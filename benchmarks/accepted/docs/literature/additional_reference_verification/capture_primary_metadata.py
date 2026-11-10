"""Bounded primary metadata capture; no manuscript or study numerical reads."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import datetime, hashlib, json, requests

ROOT = Path(__file__).resolve().parent
SOURCES = [
 ('barnston_crossref.json', 'https://api.crossref.org/works/10.1080%2F07055900.1997.9649597'),
 ('ersst6_part1_crossref.json', 'https://api.crossref.org/works/10.1175%2FJCLI-D-23-0707.1'),
 ('ersst6_part2_crossref.json', 'https://api.crossref.org/works/10.1175%2FJCLI-D-24-0185.1'),
 ('mei_v2_zhang_crossref.json', 'https://api.crossref.org/works/10.1029%2F2019GL083946'),
 ('merra2_gelaro_crossref.json', 'https://api.crossref.org/works/10.1175%2FJCLI-D-16-0758.1'),
 ('gldas21_datacite.json', 'https://api.datacite.org/dois/10.5067/E7TYRXPJKWOQ'),
 ('gldas21_cmr.json', 'https://cmr.earthdata.nasa.gov/search/collections.json?short_name=GLDAS_NOAH025_3H&version=2.1'),
 ('ncei_ersst.html', 'https://www.ncei.noaa.gov/products/extended-reconstructed-sst'),
 ('noaa_mei_v2.html', 'https://psl.noaa.gov/enso/mei/'),
 ('power_referencing.html', 'https://power.larc.nasa.gov/docs/referencing/'),
 ('power_acknowledgements.html', 'https://power.larc.nasa.gov/docs/acknowledgements/'),
 ('nasa_gmao_merra2.html', 'https://gmao.gsfc.nasa.gov/gmao-products/merra-2/documentation_merra-2/'),
 ('nasa_srb_acknowledgement.html', 'https://science.larc.nasa.gov/gewex-srb/data-usage-acknowledgement/'),
]

def fetch(item):
    filename, url = item
    receipt = {'filename': 'raw/'+filename, 'requested_url': url,
               'retrieved_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        r = requests.get(url, timeout=(15,40), headers={'User-Agent':'Scientific-reference-metadata-verification/1.0'})
        receipt.update(status_code=r.status_code, final_url=r.url,
                       content_type=r.headers.get('Content-Type'), bytes=len(r.content),
                       sha256=hashlib.sha256(r.content).hexdigest())
        (ROOT/'raw'/filename).write_bytes(r.content)
        if r.status_code != 200:
            receipt['verification_status']='HTTP_FAILURE_PRESERVED'
        else:
            receipt['verification_status']='RETRIEVED'
    except Exception as exc:
        receipt.update(verification_status='ACCESS_FAILURE',error=type(exc).__name__+': '+str(exc))
    return receipt

if __name__ == '__main__':
    (ROOT/'raw').mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts=list(pool.map(fetch,SOURCES))
    (ROOT/'retrieval_manifest.json').write_text(json.dumps(receipts,indent=2)+'\n',encoding='utf-8')
    for r in receipts:
        print(r['filename'],r['verification_status'],r.get('status_code'),r.get('bytes'))
