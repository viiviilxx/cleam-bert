import torch, random, os, gc, importlib
import torch.nn as nn
import numpy as np
from model import Model
from data_helper import BertHelper, load_label_tokens
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from tqdm import tqdm

if importlib.util.find_spec("apex") is not None:
    enable_apex = True
    from apex import amp
else:
    enable_apex = False


class Trainer:
    def __init__(self, params):
        self.params = params
        np.random.seed(params["seed"])
        random.seed(params["seed"])
        torch.manual_seed(params["seed"])
        torch.cuda.manual_seed(params["seed"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.best_min_score = 99999999999999999

    def build(self):
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        train_helper = BertHelper(self.params["train_path"], self.params["device"])
        print("making train dataloader... ", end="", flush=True)
        train_g = torch.Generator()
        train_g.manual_seed(self.params["seed"])
        self.train_loader = DataLoader(
            train_helper,
            batch_size=self.params["batch_size"],
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=train_g,
        )
        print("done!", flush=True)

        val_helper = BertHelper(self.params["val_path"], self.params["device"])
        print("making valid dataloader... ", end="", flush=True)
        val_g = torch.Generator()
        val_g.manual_seed(self.params["seed"])
        self.val_loader = DataLoader(
            val_helper, batch_size=self.params["batch_size"], shuffle=True, worker_init_fn=seed_worker, generator=val_g
        )
        print("done!", flush=True)

        if not self.params["tuning"]:
            test_helper = BertHelper(self.params["test_path"], self.params["device"])
            print("making test dataloader... ", end="", flush=True)
            test_g = torch.Generator()
            test_g.manual_seed(self.params["seed"])
            self.test_loader = DataLoader(
                test_helper,
                batch_size=self.params["batch_size"],
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=test_g,
            )
            print("done!", flush=True)
            print(
                "dataset samples {train:"
                + str(len(train_helper))
                + ", valid:"
                + str(len(val_helper))
                + ", test:"
                + str(len(test_helper))
                + "}"
            )
        else:
            print("dataset samples {train:" + str(len(train_helper)) + ", valid:" + str(len(val_helper)) + "}")

        print("creating model... ", end="")

        label_tokens = load_label_tokens(self.params["label_token_path"])
        label_tokens["input_ids"] = label_tokens["input_ids"].to(self.params["device"])
        label_tokens["attention_mask"] = label_tokens["attention_mask"].to(self.params["device"])
        self.params["label_tokens"] = label_tokens

        self.model = Model(self.params)
        self.model.to(self.params["device"])

        self.loss_fn = nn.BCEWithLogitsLoss()

        no_decay = ["LayerNorm.weight", "bias"]
        optimizer_parameters = [
            {
                "params": [p for i, p in self.model.named_parameters() if not any(j in i for j in no_decay)],
                "weight_decay": self.params["weight_decay"],
            },
            {
                "params": [p for i, p in self.model.named_parameters() if any(j in i for j in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(optimizer_parameters, lr=self.params["learning_rate"])

        num_training_steps = len(self.train_loader) * self.params["epoch"]
        num_warmup_steps = int(num_training_steps * self.params["warmup_scale"])
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)

        self.use_apex = False
        if self.params["opt_level"] is not None:
            if enable_apex:
                self.model, self.optimizer = amp.initialize(
                    self.model, self.optimizer, opt_level=self.params["opt_level"]
                )
                self.use_apex = True
            else:
                raise Exception("can not use apex")

        print("done!")

    def run(self, trial=None):
        if self.params["tuning"]:
            print("selecting hyper-parameter...", end="", flush=True)
            learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
            epoch = trial.suggest_int("epoch", 3, 15)
            learning_params = {
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "epoch": epoch,
            }
            self.params.update(learning_params)
            print("done!")

        self.build()

        print("training start")

        for epoch in range(1, self.params["epoch"] + 1):
            self.train(epoch)
            result = self.test("val")

        if not self.params["tuning"]:
            print("Finished Optimization!")
            result = self.test("test")
            loss = result["loss"]
            torch.save(self.model.state_dict(), "./saved_models/model.pkl")
        else:
            loss = result["loss"]
            if loss < self.best_min_score:
                self.best_min_score = loss
                if os.path.exists("./saved_models/tuned_model.pkl"):
                    os.remove("./saved_models/tuned_model.pkl")
                torch.save(self.model.state_dict(), "./saved_models/tuned_model.pkl")

        return loss

    def train(self, epoch):
        loss_str = "loss={:.4f}"
        self.model.train()

        with tqdm(enumerate(self.train_loader), total=len(self.train_loader), dynamic_ncols=True) as bar:
            for i, (texts, masks, labels) in bar:
                outputs = self.model(texts, masks)
                train_loss = self.loss_fn(outputs, labels)

                if self.use_apex:
                    with amp.scale_loss(train_loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    train_loss.backward()

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                bar.set_description("Train epoch %d" % epoch)
                bar.set_postfix_str(loss_str.format(train_loss.item()))

    def test(self, mode):
        self.model.eval()

        y_pre = []
        y_true = []

        if mode == "val":
            loader = self.val_loader
        else:
            loader = self.test_loader

        with tqdm(enumerate(loader), total=len(loader), dynamic_ncols=True) as bar:
            result = {}
            total_loss = []
            loss_str = "loss={:.4f}"

            for i, (texts, masks, labels) in bar:
                with torch.no_grad():
                    outputs = self.model(texts, masks)
                    test_loss = self.loss_fn(outputs, labels)
                    outputs = torch.sigmoid(outputs)

                outputs = outputs.cpu()
                outputs = outputs > self.params["threshold"]
                y_pre.extend(outputs.tolist())

                labels = labels.cpu()
                y_true.extend(labels.tolist())

                test_loss = test_loss.cpu()
                total_loss.append(test_loss.tolist())

                bar.set_description(mode)
                bar.set_postfix_str(loss_str.format(test_loss))

        y_pre = np.array(y_pre, dtype=np.float)
        y_true = np.array(y_true, dtype=np.float)
        total_loss = np.array(total_loss, dtype=np.float)

        loss_mean = np.mean(total_loss)
        print("loss/val : " + str(loss_mean))
        result["loss"] = loss_mean

        macroF1 = f1_score(y_true, y_pre, average="macro", zero_division=0)
        microF1 = f1_score(y_true, y_pre, average="micro", zero_division=0)
        print("macro f1 : " + str(macroF1))
        print("micro f1 : " + str(microF1))
        result["macro f1"] = macroF1
        result["micro f1"] = microF1

        return result

    def after_tuning(self, best_params):
        self.params.update(best_params)
        self.params["tuning"] = False
        self.test_tuned_model("tuned_model.pkl")

    def test_tuned_model(self, model_name):
        self.build()
        self.model.load_state_dict(torch.load("./saved_models/" + model_name))
        self.test("test")
