import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

from ts3.train.infra import (
    prepare_config,
    register_resolvers,
    setup_training_infra,
)


register_resolvers()


def assert_not_hydra_sweep() -> None:
    hydra_config = HydraConfig.get()
    if hydra_config.mode.name != "RUN":
        raise ValueError(
            "Hydra multirun/sweep mode is disabled for ts3. "
            "Use ts3/train/scripts/run_slide_embedding.sh or Slurm array jobs "
            "with Hydra overrides instead."
        )


def build_dataloader(
    split_cf: DictConfig,
    infra_cf=None,
    accumulate_grad_batches=None,
):
    dataset = instantiate(split_cf.dataset)
    loader_cf = split_cf.dataloader
    sampler_cf = split_cf.sampler

    if sampler_cf is not None:
        sampler_args = dict(
            sample_count=len(dataset),
            samples_per_epoch=(
                split_cf.steps_per_epoch
                * accumulate_grad_batches
                * loader_cf.batch_size
                * infra_cf.world_size
            ),
            seed=infra_cf.seed,
            rank=infra_cf.rank,
            world_size=infra_cf.world_size,
        )
        sampler = instantiate(sampler_cf)(**sampler_args)
    else:
        sampler = None

    return instantiate(loader_cf)(
        dataset=dataset,
        sampler=sampler,
    )


@hydra.main(version_base=None, config_path=".", config_name="config/slide_embedding")
def main(cf: DictConfig):
    assert_not_hydra_sweep()
    cf = prepare_config(cf)
    setup_training_infra(cf)

    train_dataloader = build_dataloader(
        cf.data.splits.train,
        infra_cf=cf.infra,
        accumulate_grad_batches=cf.trainer.accumulate_grad_batches,
    )
    val_dataloader = build_dataloader(cf.data.splits.trainval)

    model = instantiate(cf.meta_arch.model)
    lightning_module = instantiate(cf.meta_arch, model=model)
    trainer = instantiate(cf.trainer)
    trainer.fit(
        lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    main()
