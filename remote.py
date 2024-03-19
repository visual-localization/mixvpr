import pathlib
import modal

stub = modal.Stub(name="MixVPR")

volume = modal.Volume.from_name("sfxl-small")

p = pathlib.Path("")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .run_commands("git clone git@github.com:amaralibey/MixVPR.git && cd MixVPR")
    .run_commands("pip install -r requirements.txt")
)


@stub.function(
    mounts=[
        modal.Mount.from_local_file(
            "./LOGS/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt",
            remote_path="/LOGS/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt",
        )
    ]
    volumes={""}
)
def main():
    pass
