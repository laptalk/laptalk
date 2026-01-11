# LapTalk Troubleshooting Guide

## KVM/QEMU VM Microphone Passthrough

### Symptoms
- Recording creates files of correct size but with no audio content
- `arecord` works on host but VM recordings are silent
- laptalk returns truncated/garbled transcriptions

### Quick Checks

**1. Check VM capture volume (often resets to 0% on reboot):**
```bash
# In VM
amixer -c 0 sget Capture
# If at 0%, set to 80%:
amixer -c 0 sset Capture 80%
```

**2. Make volume persistent:**
```bash
sudo alsactl store
```

**3. Test recording:**
```bash
arecord -f S16_LE -r 16000 -c 1 -d 5 test.wav
# Copy to host and play to verify audio content
```

---

## Working VM Audio Configuration

### Recommended: PulseAudio Passthrough

In VM XML (`virsh edit <vm-name>`), use:
```xml
<audio id='1' type='pulseaudio' serverName='/run/user/1000/pulse/native'/>
```

Note: Change `1000` to your user ID on the host (`id -u`).

### Why Not Spice Audio?
Spice audio input passthrough is unreliable. The Spice client (virt-manager) often doesn't create capture streams from the host microphone, resulting in silent recordings in the VM.

### Alternative: PipeWire Passthrough
```xml
<audio id="1" type="pipewire" runAsUser="yes">
  <input mixingEngine="no"/>
  <output mixingEngine="no"/>
</audio>
```

The `runAsUser="yes"` is critical - it tells QEMU to access audio as your user instead of libvirt-qemu.

---

## Host Audio Issues

### Symptoms
- No audio devices in `wpctl status` (only "Dummy Output")
- `pactl list sources` shows no microphones
- GNOME Sound Settings shows no input devices

### Root Cause
Installing `pulseaudio` package conflicts with and removes:
- `pipewire-audio`
- `pipewire-alsa`

These packages are critical for PipeWire to interface with ALSA hardware.

### Fix
```bash
# Remove conflicting pulseaudio (pipewire-pulse provides compatibility)
sudo apt remove pulseaudio

# Install required PipeWire packages
sudo apt install pipewire-audio pipewire-alsa

# Restart audio stack
systemctl --user restart pipewire pipewire-pulse wireplumber

# If devices still don't appear, REBOOT
```

### Verify Host Audio
```bash
# Check PipeWire sees audio devices
wpctl status | grep -A 10 "Audio"

# Check microphones are available
pactl list sources | grep -A 3 "Name.*input"

# Verify ALSA sees hardware
aplay -l
arecord -l
```

---

## Debugging Commands

### On Host
```bash
# Check if VM is capturing from host mic
pactl list source-outputs

# Check PipeWire status
wpctl status

# Check recent package changes (to find what broke)
grep "$(date +%Y-%m-%d)" /var/log/dpkg.log | grep -E 'pipewire|pulse|alsa'

# Check WirePlumber logs
journalctl --user -u wireplumber --since "10 minutes ago"
```

### On VM
```bash
# Check PipeWire sources
wpctl status | grep -A 5 "Sources"

# Check ALSA capture settings
amixer -c 0 sget Capture

# List recording devices
arecord -l
```

---

## Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| VM recordings are silent | Capture volume at 0% | `amixer -c 0 sset Capture 80%` |
| Volume resets on reboot | Settings not saved | `sudo alsactl store` |
| Host has no audio devices | pipewire-audio/alsa removed | Reinstall packages + reboot |
| Spice audio not capturing | Spice client limitation | Switch to PulseAudio passthrough |
| QEMU permission denied | Missing runAsUser | Add `runAsUser="yes"` to audio XML |

---

## Package Dependencies (Ubuntu 24.04)

### Required on Host for VM Audio Passthrough
- `pipewire`
- `pipewire-pulse`
- `pipewire-audio`
- `pipewire-alsa`
- `wireplumber`
- `libspa-0.2-modules` (provides ALSA SPA plugin)

### Conflicting Packages (will break PipeWire)
- `pulseaudio` (use `pipewire-pulse` instead)

### Check Installation
```bash
dpkg -l | grep -E 'pipewire-audio|pipewire-alsa'
# Both should be listed as 'ii' (installed)
```
