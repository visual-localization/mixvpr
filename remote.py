import modal


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .run_commands("git clone https://github.com/visual-localization/mixvpr.git")
    .pip_install_from_requirements("requirements.txt")
    .workdir("/root/mixvpr")
)

stub = modal.Stub(
    image=image,
    mounts=[modal.Mount.from_local_dir("/LOGS", remote_path="/root/mixvpr/LOGS")],
)


@stub.local_entrypoint()
def entry():
    from main import main

    main()
