#!/usr/bin/env python

import os, sys
import subprocess

ffm = "/usr/local/bin/ffmpeg"

def run(cmd, check=False):
    p = subprocess.run(cmd, shell=1, check=check, )
    if p.stderr != '':
        print(p.stderr)
        sys.exit()
    return p.stdout, p.stderr

def cleanup(fn):
    if os.path.exists(fn):
        os.system(f"rm -f {fn}")

def make_gif(images=None, txt=None, fr=3, fo="movie", fmt="avi",
             directgif=0):
    if os.path.exists(f"{fo}.avi"):
        print(f"{fo}.avi exists. Skipped")
        return
    if os.path.exists(f"{fo}.gif"):
        print(f"{fo}.gif exists. Skipped")
        return
    if txt is None:
        assert images is not None
        run(f"{ffm} -r {fr} -start_number 0 -f image2 -i {images} -vcodec "
            f"mjpeg -qscale 1 {fo}.avi")
        run(f"{ffm} -i {fo}.avi -pix_fmt rgb24 {fo}.gif")
    else:
        if fmt == "gif" and directgif:
            cleanup(f"{fo}.gif")
            run(f"{ffm} -f concat -safe 0 -r {fr} -i {txt} {fo}.gif")
            return
        if os.path.exists(f"{fo}.avi"):
            print(f"{fo}.avi exists. Skipped")
            return
        run(f"{ffm} -f concat -safe 0 -r {fr} -i {txt} -vcodec libx264 -pix_fmt yuv420p"
            f" -qscale 1 {fo}.mp4")
        if fmt == "gif":
            cleanup(f"{fo}.gif")
            run(f"{ffm} -i {fo}.avi -pix_fmt rgb24 {fo}.gif")

def main():
    return

if __name__ == "__main__":

    main()
