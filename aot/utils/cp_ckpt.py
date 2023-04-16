import os
import shutil


def cp_ckpt(remote_dir="data_wd/youtube_vos_jobs/result", curr_dir="backup"):
    exps = os.listdir(curr_dir)
    for exp in exps:
        print("Exp: ", exp)
        exp_dir = os.path.join(curr_dir, exp)
        stages = os.listdir(exp_dir)
        for stage in stages:
            print("Stage: ", stage)
            stage_dir = os.path.join(exp_dir, stage)
            finals = ["ema_ckpt", "ckpt"]
            for final in finals:
                print("Final: ", final)
                final_dir = os.path.join(stage_dir, final)
                ckpts = os.listdir(final_dir)
                for ckpt in ckpts:
                    if '.pth' not in ckpt:
                        continue
                    curr_ckpt_path = os.path.join(final_dir, ckpt)
                    remote_ckpt_path = os.path.join(remote_dir, exp, stage,
                                                    final, ckpt)
                    if os.path.exists(remote_ckpt_path):
                        os.system('rm {}'.format(remote_ckpt_path))
                    try:
                        shutil.copy(curr_ckpt_path, remote_ckpt_path)
                        print(ckpt, ': OK')
                    except OSError as Inst:
                        print(Inst)
                        print(ckpt, ': Fail')


if __name__ == "__main__":
    cp_ckpt()
