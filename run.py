import optuna, argparse, torch, json
from train import Trainer
from transformers import logging
from transformers import BertConfig

logging.set_verbosity_error()


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["rcv1v2", "wos", "nyt"], default="rcv1v2")
parser.add_argument("--tuning", action="store_true", default=False)
parser.add_argument("--opt_level", choices=[None, "O0", "O1", "O2"], default="O2")
parser.add_argument("--test_tuned_model", action="store_true", default=False)


def main():
    args = parser.parse_args()
    args = vars(args)

    with open("./config/" + args["dataset"] + "_config.json", "r") as f:
        params = json.load(f)
    params.update(args)

    if torch.cuda.is_available():
        params["device"] = torch.device("cuda:0")
    else:
        raise ("can't use cuda")

    params["bert_config"] = BertConfig()

    if params["tuning"]:
        trainer = Trainer(params)
        study = optuna.create_study(
            pruner=optuna.pruners.HyperbandPruner(reduction_factor=3),
        )
        hour = 3600
        study.optimize(trainer.run, timeout=hour * 24 * 2)
        print(study.best_trial)
        trainer.after_tuning(study.best_params)
    else:
        trainer = Trainer(params)
        if params["test_tuned_model"]:
            trainer.test_tuned_model("tuned_model.pkl")
        else:
            trainer.run()


if __name__ == "__main__":
    main()
