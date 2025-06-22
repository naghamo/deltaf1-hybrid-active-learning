from adaptive_al.models import get_model
from transformers import Trainer, TrainingArguments, set_seed
from adaptive_al.utils.trainer import evaluate_model
import torch
import random
import numpy as np

class RetrainStrategy:
    def __init__(self, config, model_name, num_labels):
        self.model_name = model_name
        self.num_labels = num_labels
        self.config = config

        if "seed" in config:
            seed = config["seed"]
            set_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def train(self, train_dataset, val_dataset):
        model = get_model(self.model_name, self.num_labels)

        args = TrainingArguments(
            output_dir=self.config["output_dir"],
            num_train_epochs=self.config["epochs"],
            per_device_train_batch_size=self.config["batch_size"],
            learning_rate=self.config["learning_rate"],
            warmup_steps=self.config.get("warmup_steps", 0),
            weight_decay=self.config.get("weight_decay", 0.0),
            eval_steps=self.config.get("eval_steps", 50),
            save_strategy="no",
            logging_steps=10,
            disable_tqdm=False,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=evaluate_model,
        )

        trainer.train()
        eval_metrics = trainer.evaluate()
        return model, eval_metrics
