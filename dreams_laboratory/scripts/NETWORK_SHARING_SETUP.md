# Network Sharing Setup Guide

## Quick Setup Options

### Option 1: **SSH/SCP** (Simplest, Already Available)

**Best for**: File transfer, remote access

```bash
# Test connection
ssh user@192.168.0.232

# Copy files TO remote machine
scp file.txt user@192.168.0.232:/path/to/destination/

# Copy files FROM remote machine
scp user@192.168.0.232:/path/to/file.txt ./

# Copy entire directories
scp -r directory/ user@192.168.0.232:/path/to/destination/

# Mount remote filesystem via SSHFS
sudo apt-get install sshfs
mkdir ~/remote_mount
sshfs user@192.168.0.232:/path/to/share ~/remote_mount
# Now access files as if local
# Unmount: fusermount -u ~/remote_mount
```

---

### Option 2: **NFS (Network File System)** (Best for Linux-to-Linux)

**Best for**: Large datasets, persistent mounts, high performance

#### On Server Machine (192.168.0.232):

```bash
# Install NFS server
sudo apt-get update
sudo apt-get install nfs-kernel-server

# Create/share directory
sudo mkdir -p /shared
sudo chmod 777 /shared  # Or set proper permissions

# Edit exports file
sudo nano /etc/exports

# Add this line (replace YOUR_IP with actual IP):
/shared 192.168.0.0/24(rw,sync,no_subtree_check)

# Or allow specific IP:
/shared 192.168.0.XXX(rw,sync,no_subtree_check)

# Restart NFS service
sudo exportfs -ra
sudo systemctl restart nfs-kernel-server

# Check status
sudo systemctl status nfs-kernel-server
```

#### On Client Machine (This Machine):

```bash
# Install NFS client
sudo apt-get install nfs-common

# Create mount point
sudo mkdir -p /mnt/shared

# Mount the share
sudo mount -t nfs 192.168.0.232:/shared /mnt/shared

# Test
ls /mnt/shared

# Make permanent (add to /etc/fstab)
sudo nano /etc/fstab
# Add this line:
192.168.0.232:/shared /mnt/shared nfs defaults 0 0

# Test fstab entry
sudo mount -a
```

**Mount Options for Better Performance**:
```bash
sudo mount -t nfs -o rw,sync,hard,intr,rsize=8192,wsize=8192,timeo=14 \
  192.168.0.232:/shared /mnt/shared
```

---

### Option 3: **Samba/SMB** (Cross-Platform)

**Best for**: Windows compatibility, easier setup

#### On Server Machine (192.168.0.232):

```bash
# Install Samba
sudo apt-get install samba

# Configure
sudo nano /etc/samba/smb.conf

# Add at end:
[shared]
   path = /shared
   browseable = yes
   writable = yes
   valid users = your_username
   create mask = 0664
   directory mask = 0775

# Set Samba password
sudo smbpasswd -a your_username

# Restart
sudo systemctl restart smbd
sudo systemctl enable smbd
```

#### On Client Machine (This Machine):

```bash
# Install Samba client
sudo apt-get install cifs-utils

# Create mount point
sudo mkdir -p /mnt/shared

# Mount
sudo mount -t cifs //192.168.0.232/shared /mnt/shared \
  -o username=your_username,password=your_password,uid=$(id -u),gid=$(id -g)

# Or use credentials file (more secure)
sudo nano ~/.smbcredentials
# Add:
username=your_username
password=your_password
domain=workgroup

sudo chmod 600 ~/.smbcredentials

# Mount with credentials
sudo mount -t cifs //192.168.0.232/shared /mnt/shared \
  -o credentials=~/.smbcredentials,uid=$(id -u),gid=$(id -g)

# Make permanent (add to /etc/fstab)
//192.168.0.232/shared /mnt/shared cifs credentials=~/.smbcredentials,uid=1000,gid=1000 0 0
```

---

### Option 4: **rsync** (Efficient Syncing)

**Best for**: One-time or periodic syncing

```bash
# Sync TO remote
rsync -avz --progress /local/path/ user@192.168.0.232:/remote/path/

# Sync FROM remote
rsync -avz --progress user@192.168.0.232:/remote/path/ /local/path/

# Sync large datasets (with resume capability)
rsync -avz --progress --partial /mnt/22tb-hdd/ user@192.168.0.232:/backup/

# Dry run (see what would be synced)
rsync -avz --dry-run /local/path/ user@192.168.0.232:/remote/path/
```

---

## Recommended Setup for Your Use Case

Given you're working with **large multispectral datasets** (22TB drive), I recommend:

### **NFS for Shared Storage**

```bash
# On 192.168.0.232 (server):
sudo apt-get install nfs-kernel-server
sudo mkdir -p /shared/datasets
sudo chmod 777 /shared/datasets  # Adjust permissions as needed

sudo nano /etc/exports
# Add:
/shared/datasets 192.168.0.0/24(rw,sync,no_subtree_check,no_root_squash)

sudo exportfs -ra
sudo systemctl restart nfs-kernel-server

# On this machine (client):
sudo apt-get install nfs-common
sudo mkdir -p /mnt/remote_datasets
sudo mount -t nfs -o rw,sync,hard,intr,rsize=8192,wsize=8192 \
  192.168.0.232:/shared/datasets /mnt/remote_datasets

# Add to /etc/fstab for permanent mount:
sudo nano /etc/fstab
# Add line:
192.168.0.232:/shared/datasets /mnt/remote_datasets nfs rw,sync,hard,intr,rsize=8192,wsize=8192 0 0
```

---

## Performance Tips

### For Large Files (Multispectral TIFFs):

1. **Use NFS with optimized options**:
```bash
mount -t nfs -o rw,sync,hard,intr,rsize=131072,wsize=131072,timeo=600 \
  192.168.0.232:/shared /mnt/shared
```

2. **Consider 10GbE if available** (faster than 1GbE)

3. **Use rsync for initial sync, NFS for ongoing access**

---

## Security Considerations

1. **Restrict access**:
```bash
# In /etc/exports, use specific IP:
/shared 192.168.0.XXX(rw,sync,no_subtree_check)
```

2. **Firewall rules**:
```bash
# Allow NFS (if using firewall)
sudo ufw allow from 192.168.0.0/24 to any port nfs
```

3. **SSH keys** (for SSH/SCP):
```bash
# Generate key pair
ssh-keygen -t rsa

# Copy to remote
ssh-copy-id user@192.168.0.232

# Now passwordless SSH
```

---

## Quick Test Commands

```bash
# Test connectivity
ping -c 3 192.168.0.232

# Test SSH
ssh user@192.168.0.232 "echo 'Connection successful'"

# Test NFS
showmount -e 192.168.0.232

# Test Samba
smbclient -L //192.168.0.232 -U username

# Check mount
df -h | grep shared
mount | grep shared
```

---

## Troubleshooting

### NFS Issues:
```bash
# Check exports
sudo exportfs -v

# Check NFS status
sudo systemctl status nfs-kernel-server

# Check mount
mount | grep nfs

# Force unmount if stuck
sudo umount -l /mnt/shared
```

### Samba Issues:
```bash
# Check Samba status
sudo systemctl status smbd

# Test configuration
sudo testparm

# Check shares
smbclient -L //192.168.0.232 -U username
```

---

## Recommendation

**For your use case** (large datasets, Linux-to-Linux):

1. **Start with SSH/SSHFS** - Quick and easy, already available
2. **Move to NFS** - Better performance for large files, persistent mounts
3. **Use rsync** - For periodic backups/syncing

The best approach depends on your specific needs:
- **Temporary access**: SSHFS
- **Persistent shared storage**: NFS
- **One-time sync**: rsync
- **Cross-platform**: Samba

