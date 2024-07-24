from ts2.train.main import instantiate_lightning_module
import yaml

cf = yaml.safe_load("""
- which: UNIEvalSystem
  params:
    ckpt_path: /nfs/mm-isilon/brainscans/dropbox/exp/models/uni/vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin
    tag: uni
- which: ConchEvalSystem
  params:
    ckpt_path: /nfs/mm-isilon/brainscans/dropbox/exp/models/conch/pytorch_model.bin
    tag: conch
- which: VirchowEvalSystem
  params:
    ckpt_path: /nfs/mm-isilon/brainscans/dropbox/exp/models/virchow/pytorch_model.bin
    tag: virchow
- which: PLIPEvalSystem
  params:
    ckpt_path: /nfs/mm-isilon/brainscans/dropbox/exp/models/plip/
    tag: plip
- which: GigapathEvalSystem
  params:
    ckpt_path: /nfs/mm-isilon/brainscans/dropbox/exp/models/gigapath/pytorch_model.bin
    tag: gigapath
""")

for cf_ in cf:
    print(cf_["params"]["tag"])
    del cf_["params"]["tag"]
    model = instantiate_lightning_module(                              
        which=cf_["which"],
        params=cf_["params"],
        training_params=None)                                               
    print(sum(p.numel() for p in model.model.parameters()))
