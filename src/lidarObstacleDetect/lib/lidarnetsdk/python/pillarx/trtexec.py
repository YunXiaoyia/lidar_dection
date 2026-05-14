from . import PILLARX_PACKAGE_FOLDER

if __name__ == "__main__":
    import argparse
    import subprocess
    from pathlib import Path

    trtexec = Path(PILLARX_PACKAGE_FOLDER) / "trtexec"
    assert trtexec.exists(
    ), f"trtexec not availble in {trtexec}, check your installation"

    trtexec_plugins = Path(PILLARX_PACKAGE_FOLDER) / "libtrtexec_plugins.so"
    assert trtexec_plugins.exists(
    ), f"trtexec_plugins not availble in {trtexec_plugins}, check your installation"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx",
        type=Path,
        help="onnx path",
        required=True,
    )
    parser.add_argument(
        "--saveEngine",
        type=Path,
        help="engine export path",
        required=True,
    )
    parser.add_argument(
        "--workspace",
        type=int,
        help="tensorrt workspace",
        required=True,
    )
    args = parser.parse_args()

    subprocess.check_call([
        str(trtexec),
        f"--onnx={args.onnx}",
        f"--saveEngine={args.saveEngine}",
        f"--plugins={trtexec_plugins}",
        f"--workspace={args.workspace}",
    ])