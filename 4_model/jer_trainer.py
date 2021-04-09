from transformers.trainer import *

class JERTrainer(Trainer):
    def _prediction_loop(
            self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
        ) -> PredictionOutput:
            """
            Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

            Works both with or without labels.
            """

            prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only

            model = self.model
            # multi-gpu eval
            if self.args.n_gpu > 1:
                model = torch.nn.DataParallel(model)
            else:
                model = self.model
            # Note: in torch.distributed mode, there's no point in wrapping the model
            # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

            batch_size = dataloader.batch_size
            logger.info("***** Running %s *****", description)
            logger.info("  Num examples = %d", self.num_examples(dataloader))
            logger.info("  Batch size = %d", batch_size)
            eval_losses: List[float] = []
            preds: torch.Tensor = None
            preds_relations: torch.Tensor = None
            label_ids: torch.Tensor = None
            model.eval()
            # to support the joint recognition relation, both preds and preds_relations exist. the logits
            # variable is actually a list, with preds = logits[0], and preds_relations = logits[1]
            if is_torch_tpu_available():
                dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

            if self.args.past_index >= 0:
                past = None

            for inputs in tqdm(dataloader, desc=description):
                has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])

                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.args.device)
                if self.args.past_index >= 0:
                    inputs["mems"] = past

                with torch.no_grad():
                    outputs = model(**inputs)
                    if has_labels:
                        step_eval_loss, logits = outputs[:2]
                        eval_losses += [step_eval_loss.mean().item()]
                    else:
                        logits = outputs[0]
                    if self.args.past_index >= 0:
                        past = outputs[self.args.past_index if has_labels else self.args.past_index - 1]

                if not prediction_loss_only:
                    if preds is None:
                        preds = logits[0].detach()
                    else:
                        preds = torch.cat((preds, logits[0].detach()), dim=0)
                    if preds_relations is None:
                        preds_relations = logits[1].detach()
                    else:
                        preds_relations = torch.cat((preds_relations, logits[1].detach()), dim=0)
                    if inputs.get("labels") is not None:
                        if label_ids is None:
                            label_ids = inputs["labels"].detach()
                        else:
                            label_ids = torch.cat((label_ids, inputs["labels"].detach()), dim=0)

            if self.args.local_rank != -1:
                # In distributed mode, concatenate all results from all nodes:
                if preds is not None:
                    preds = self.distributed_concat(preds, num_total_examples=self.num_examples(dataloader))
                if label_ids is not None:
                    label_ids = self.distributed_concat(label_ids, num_total_examples=self.num_examples(dataloader))
            elif is_torch_tpu_available():
                # tpu-comment: Get all predictions and labels from all worker shards of eval dataset
                if preds is not None:
                    preds = xm.mesh_reduce("eval_preds", preds, torch.cat)
                if label_ids is not None:
                    label_ids = xm.mesh_reduce("eval_label_ids", label_ids, torch.cat)

            # Finally, turn the aggregated tensors into numpy arrays.
            if preds is not None:
                preds = preds.cpu().numpy()
            if preds_relations is not None:
                preds_relations = preds_relations.cpu().numpy()
            if label_ids is not None:
                label_ids = label_ids.cpu().numpy()

            if self.compute_metrics is not None and preds is not None and label_ids is not None:
                metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
            else:
                metrics = {}
            if len(eval_losses) > 0:
                metrics["eval_loss"] = np.mean(eval_losses)

            # Prefix all keys with eval_
            for key in list(metrics.keys()):
                if not key.startswith("eval_"):
                    metrics[f"eval_{key}"] = metrics.pop(key)

            return PredictionOutput(predictions=(preds, preds_relations), label_ids=label_ids, metrics=metrics)
