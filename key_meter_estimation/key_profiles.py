"""
An assorted collection of Key Profiles
"""
import numpy as np
from scipy.linalg import circulant

### Definitions

# List of labels for each key (Use enharmonics as needed)
KEYS = ["C","C#","D","Eb","E","F","Gb","G","Ab","A","Bb","B",
        "Cm","C#m","Dm","D#m","Em","Fm","F#m","Gm","G#m","Am","A#m","Bm"]

PITCH_CLASSES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

MAJOR_KEYS = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

MINOR_KEYS = ["Cm","C#m","Dm","D#m","Em","Fm","F#m","Gm","G#m","Am","A#m","Am"]

MAJ_min_KEYS = [ky for kk in zip(MAJOR_KEYS,MINOR_KEYS) for ky in kk]

## Krumhansl Kessler Key Profiles

# From Krumhansl's "Cognitive Foundations of Musical Pitch" pp.30
key_prof_maj_kk = np.array(
    [6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])

key_prof_min_kk = np.array(
    [6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

## Temperley Key Profiles

# CBMS (from "Music and Probability" Table 6.1, pp. 86)
key_prof_maj_cbms = np.array(
    [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0])

key_prof_min_cbms = np.array(
    [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0])

# Kostka-Payne (from "Music and Probability" Table 6.1, pp. 86)

key_prof_maj_kp = np.array(
    [0.748, 0.060, 0.488, 0.082, 0.670, 0.460, 0.096, 0.715, 0.104, 0.366, 0.057, 0.400])

key_prof_min_kp = np.array(
    [0.712, 0.048, 0.474, 0.618, 0.049, 0.460, 0.105, 0.747, 0.404, 0.067, 0.133, 0.330])

def build_key_matrix():
    maj_key_matrix = np.array(MAJOR_KEYS)
    min_key_matrix = np.array(MINOR_KEYS)

    for ii in range(1,12):
        maj_key_matrix = np.vstack((maj_key_matrix,np.roll(MAJOR_KEYS,-ii)))
        min_key_matrix = np.vstack((min_key_matrix,np.roll(MINOR_KEYS,-ii)))
        
    return maj_key_matrix,min_key_matrix

def _build_key_profile_matrix(key_prof_maj,key_prof_min):
    
    # Normalize Key profiles
    key_prof_maj /= np.sum(key_prof_maj)
    key_prof_min /= np.sum(key_prof_min)

    # Create matrix of key profiles
    Key_prof_mat = np.vstack(
        (circulant(key_prof_maj).transpose(),
         circulant(key_prof_min).transpose()))
    
    return Key_prof_mat

def build_key_profile_matrix(profile='kk'):
    
    assert profile in ['kk','cbms','kp']
    
    if profile == 'kk':
        key_prof_maj = key_prof_maj_kk
        key_prof_min = key_prof_min_kk
    elif profile == 'cbms':
        key_prof_maj = key_prof_maj_cbms
        key_prof_min = key_prof_min_cbms
    elif profile == 'kp':
        key_prof_maj = key_prof_maj_kp
        key_prof_min = key_prof_min_kp

    Key_prof_mat = _build_key_profile_matrix(
        key_prof_maj, key_prof_min)

    return Key_prof_mat

if __name__ == "__main__":
    kpm = build_key_profile_matrix()
