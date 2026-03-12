import os
import shutil


def cp_ckpt(remote_dir="data_wd/youtube_vos_jobs/result", curr_dir="backup"):
    if not os.path.isdir(curr_dir):
        print(f"Directory not found: {curr_dir}")
        return

    for exp in os.listdir(curr_dir):
        print("Exp:", exp)
        exp_dir = os.path.join(curr_dir, exp)

        if not os.path.isdir(exp_dir):
            continue

        for stage in os.listdir(exp_dir):
            print("Stage:", stage)
            stage_dir = os.path.join(exp_dir, stage)

            if not os.path.isdir(stage_dir):
                continue

            for final in ["ema_ckpt", "ckpt"]:
                print("Final:", final)
                final_dir = os.path.join(stage_dir, final)

                if not os.path.isdir(final_dir):
                    continue

                for ckpt in os.listdir(final_dir):

                    if not ckpt.endswith(".pth"):
                        continue

                    curr_ckpt_path = os.path.join(final_dir, ckpt)
                    remote_ckpt_path = os.path.join(
                        remote_dir, exp, stage, final, ckpt
                    )

                    # ensure destination directory exists
                    os.makedirs(os.path.dirname(remote_ckpt_path), exist_ok=True)

                    # remove existing checkpoint if present
                    if os.path.exists(remote_ckpt_path):
                        try:
                            os.remove(remote_ckpt_path)
                        except OSError as e:
                            print(f"Failed to remove {remote_ckpt_path}: {e}")
                            continue

                    try:
                        shutil.copy(curr_ckpt_path, remote_ckpt_path)
                        print(ckpt, ": OK")
                    except OSError as e:
                        print(e)
                        print(ckpt, ": Fail")


if __name__ == "__main__":
    cp_ckpt()

