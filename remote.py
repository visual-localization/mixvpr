from modal import Stub, Volume,Image,Mount,gpu

from typing import Dict
import os

from const import PITTS,GSV



def lookup_volume(data_dict:Dict[str,str]):
    return dict((k, Volume.lookup(v)) for k, v in data_dict.items())
    
stub = Stub(
    name="Im trying my best :((("
)

image = (
    Image.debian_slim(python_version="3.10")
    .apt_install(["ffmpeg","libsm6","libxext6"])
    .pip_install_from_requirements("./requirements.txt")
)


vol_dict = {
    **GSV,
    **PITTS,
    "/root/LOGS": "MixVPR_LOGS"
}

@stub.function(
    image=image,
    mounts=[Mount.from_local_dir("./", remote_path="/root/mixvpr")],
    volumes=lookup_volume(vol_dict),
    _allow_background_volume_commits = True,
    gpu='a10g',
    timeout=86400
)
def entry():
    import sys
    sys.path.append("/root/mixvpr")

    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback, ModelCheckpoint
    from lightning_lite.utilities.seed import seed_everything
    import pytorch_lightning.loggers as log
    import torch
    
    from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule
    from main import VPRModel
    
    seed_everything(2011015,workers=True)
    datamodule = GSVCitiesDataModule(
        batch_size=10,
        img_per_place=8,
        min_img_per_place=16,
        shuffle_all=False,  # shuffle all images or keep shuffling in-city only
        random_sample_from_each_place=True,
        image_size=(320, 320),
        num_workers=5,
        show_data_stats=True,
        val_set_names=[
            "pitts30k_val",
            "pitts30k_test",
            #"msls_val",
        ],  # pitts30k_val, pitts30k_test, msls_val
    )

    # examples of backbones
    # resnet18, resnet50, resnet101, resnet152,
    # resnext50_32x4d, resnext50_32x4d_swsl , resnext101_32x4d_swsl, resnext101_32x8d_swsl
    # efficientnet_b0, efficientnet_b1, efficientnet_b2
    # swinv2_base_window12to16_192to256_22kft1k
    model = VPRModel(
        # ---- Encoder
        backbone_arch="resnet50",
        pretrained=True,
        layers_to_freeze=3,
        layers_to_crop=[4],  # 4 crops the last resnet layer, 3 crops the 3rd, ...etc
        # ---- Aggregator
        # agg_arch='CosPlace',
        # agg_config={'in_dim': 2048,
        #             'out_dim': 2048},
        # agg_arch='GeM',
        # agg_config={'p': 3},
        # agg_arch='ConvAP',
        # agg_config={'in_channels': 2048,
        #             'out_channels': 2048},
        agg_arch="MixVPR",
        agg_config={
            "in_channels": 1024,
            "in_h": 20,
            "in_w": 20,
            "out_channels": 1024,
            "mix_depth": 4,
            "mlp_ratio": 1,
            "out_rows": 4,
            "layers_to_freeze":1
        },  # the output dim will be (out_rows * out_channels)
        # ---- Train hyperparameters
        lr=0.01,  # 0.0002 for adam, 0.05 or sgd (needs to change according to batch size)
        optimizer="sgd",  # sgd, adamw
        weight_decay=0.001,  # 0.001 for sgd and 0 for adam,
        momentum=0.9,
        warmpup_steps=650,
        milestones=[5, 10, 15, 25, 45],
        lr_mult=0.3,
        # ----- Loss functions
        # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
        # FastAPLoss, CircleLoss, SupConLoss,
        loss_name="MultiSimilarityLoss",
        miner_name="CustomMultiSimilarityMiner",  # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
        miner_margin=0.1,
        faiss_gpu=False,
    )

    # model params saving using Pytorch Lightning
    # we save the best 3 models accoring to Recall@1 on pittsburg val
    checkpoint_cb = ModelCheckpoint(
        dirpath="/root/LOGS",
        monitor="pitts30k_val/R1",
        filename=f"{model.encoder_arch}"
        + "_epoch({epoch:02d})_step({step:04d})_R1[{pitts30k_val/R1:.4f}]_R5[{pitts30k_val/R5:.4f}]",
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        mode="max",
    )

    # ------------------
    # we instanciate a trainer
    csv_logger = log.CSVLogger(save_dir="/root/LOGS")
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        default_root_dir=f"/root/LOGS/",  # Tensorflow can be used to viz
        num_sanity_val_steps=0,  # runs a validation step before stating training
        precision=16,  # we use half precision to reduce  memory usage
        # TODO: CHange this in the future to normal epoch
        max_epochs=3,
        check_val_every_n_epoch=1,  # run validation every epoch
        callbacks=[
            checkpoint_cb
        ],  # we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1,  # we reload the dataset to shuffle the order
        log_every_n_steps=2,
        logger=csv_logger
        # fast_dev_run=True # uncomment or dev mode (only runs a one iteration train and validation, no checkpointing).
    )


    # we load the pretrained model for finetuning
    device='cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path = "/root/LOGS/init.ckpt"
    
    state_dict = torch.load(ckpt_path, map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model.to(device)

    trainer.fit(model=model, datamodule=datamodule)