import paramiko
import os
import subprocess
import os
import stat
import paramiko
import traceback

dir_path = "./result"
remote_ip = "172.24.196.6"
remote_username="guan"
remote_passward="guan"

class SSH(object):

    def __init__(self, ip, port=22, username=None, password=None, timeout=30):
        self.ip = ip
        self.port = port
        self.username = username
        self.password = password
        self.timeout = timeout

        self.ssh = paramiko.SSHClient()

        self.t = paramiko.Transport(sock=(self.ip, self.port))


    def _password_connect(self):

        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(hostname=self.ip, port=22, username=self.username, password=self.password)
        self.t.connect(username=self.username, password=self.password)

    def _key_connect(self):
        self.pkey = paramiko.RSAKey.from_private_key_file('/home/roo/.ssh/id_rsa', )
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(hostname=self.ip, port=22, username=self.username, pkey=self.pkey)
        self.t.connect(username=self.username, pkey=self.pkey)

    def connect(self):
        try:
            self._key_connect()
        except:
            print('ssh key connect failed, trying to password connect...')
            try:
                self._password_connect()
            except:
                print('ssh password connect faild!')
        
    def close(self):
        self.t.close()
        self.ssh.close()
        
    def execute_cmd(self, cmd):
        stdin, stdout, stderr = self.ssh.exec_command(cmd)
        res, err = stdout.read(), stderr.read()
        result = res if res else err
        return result.decode()

    def _sftp_get(self, remotefile, localfile):
        sftp = paramiko.SFTPClient.from_transport(self.t)
        sftp.get(remotefile, localfile)
        

            
        
    def _sftp_put(self, localfile, remotefile):
        sftp = paramiko.SFTPClient.from_transport(self.t)
        sftp.put(localfile, remotefile)

    def _get_all_files_in_remote_dir(self, sftp, remote_dir):
        all_files = list()
        if remote_dir[-1] == '/':
            remote_dir = remote_dir[0:-1]

        files = sftp.listdir_attr(remote_dir)
        for file in files:
            filename = remote_dir + '/' + file.filename

            if stat.S_ISDIR(file.st_mode): 
                all_files.extend(self._get_all_files_in_remote_dir(sftp, filename))
            else:
                all_files.append(filename)

        return all_files

    def sftp_get_file(self, file, local_filename):
        try:

            sftp = paramiko.SFTPClient.from_transport(self.t)
            sftp.get(file, local_filename)
        except:
            print('ssh get dir from master failed.')
            print(traceback.format_exc())

    def _get_all_files_in_local_dir(self, local_dir):
        all_files = list()

        for root, dirs, files in os.walk(local_dir, topdown=True):
            for file in files:
                filename = os.path.join(root, file)
                all_files.append(filename)

        return all_files

    def sftp_put_dir(self, local_dir, remote_dir):
        try:
            sftp = paramiko.SFTPClient.from_transport(self.t)

            if remote_dir[-1] == "/":
                remote_dir = remote_dir[0:-1]

            all_files = self._get_all_files_in_local_dir(local_dir)
            for file in all_files:

                remote_filename = file.replace(local_dir, remote_dir)
                remote_path = os.path.dirname(remote_filename)

                try:
                    sftp.stat(remote_path)
                except:
                    self.execute_cmd('mkdir -p %s' % remote_path) 
                print(f"put {file} to {remote_filename}")
                sftp.put(file, remote_filename)

        except:
            print('ssh get dir from master failed.')
            print(traceback.format_exc())
subps = []
def Popen(*args, **kwargs):
    p = subprocess.Popen(*args, **kwargs)
    subps.append(p)
    return p

def exec_ssh(host, command, block=False):
    args = [
        'ssh', '-o', 'ServerAliveInterval 60', '-o', 'ServerAliveCountMax 120',
        host, command,
    ]
    print(f"executing on {host}: {command}")
    p = Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if block:
        ret = p.wait()
        assert ret == 0, f'something wrong during execution: \n{p.stderr.readlines()}'
    return p


def exec_docker(host, command, block=False):
    command = f'docker exec -t adapt_nose bash -c "{command}"'
    return exec_ssh(host, command, block=block)

if __name__ == "__main__":
    paths = [p for p in os.listdir(dir_path)]
    need_test=[]
    for path in paths:
        if os.path.exists(os.path.join(os.path.join(dir_path, path), 'accuracy.csv')):
            continue
        need_test.append(path)
        print(f"{path} need to be tested")

    ssh = SSH(ip=remote_ip, username=remote_username, password=remote_passward)
    ssh.connect()
    
    print("update checkpoint models to server")
    for path in need_test:
        remotedir = os.path.join("./adapt_noise/result/", path)
        srcdir=os.path.join(dir_path, path)
        print(f"put {srcdir} into {remotedir}")
        ssh.sftp_put_dir(srcdir,remotedir)
    
    cmd = 'docker exec -t adapt_nose bash -c "cd /adapt_nose/ && python3 test_chkpts.py result/ && cd result/ && rm -rf **/chkpt"'
    print("server test checkpoint")
    print(ssh.execute_cmd(cmd))
    
    print("get accunracy result from server")
    for path in need_test:
        remotefile = os.path.join("./adapt_noise/result/", path)
        ssh.sftp_get_file(os.path.join(remotefile, 'accuracy.csv'), os.path.join(os.path.join(dir_path, path), 'accuracy.csv'))
    ssh.close()



