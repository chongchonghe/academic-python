import subprocess

def betterRun(cmd, prt=True, check=True):

    process = subprocess.run(cmd, shell=1, check=check, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,  universal_newlines=True)
    if prt:
        print(process.stdout)
    return process.stdout, process.stderr
