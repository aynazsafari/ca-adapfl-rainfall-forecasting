"""Standalone frozen ERA5 native-area helper; same arithmetic as January QA."""
import hashlib
import numpy as np
from pyproj import Geod
VARS=['t2m','d2m','sp','u10','v10','tp','msl']
OLD={'t2m':'era5_t2m','d2m':'era5_d2m','sp':'era5_sp','u10':'era5_u10','v10':'era5_v10','tp':'era5_tp','msl':'era5_mslp'}
CHUNK=(8736,12,12)
def sha(data):return hashlib.sha256(data).hexdigest()
def build_weights(mask):
    latitude=mask['latitude'];longitude=mask['longitude'];li=mask['latitude_indices'];lj=mask['longitude_indices']
    geod=Geod(ellps='WGS84');areas=[]
    for lat in latitude[li]:
        a,_=geod.polygon_area_perimeter([-.125,.125,.125,-.125],[lat-.125,lat-.125,lat+.125,lat+.125]);areas.append(abs(a))
    area=np.broadcast_to(np.asarray(areas)[:,None],(len(li),len(lj)))
    clients=[k for k in mask.files if k.startswith('Client')]
    full=np.zeros((len(clients),721,1440),dtype=np.float64)
    for i,c in enumerate(clients):full[i][np.ix_(li,lj)]=area*mask[c]
    denominators=full.sum(axis=(1,2));full/=denominators[:,None,None]
    return clients,full,denominators
