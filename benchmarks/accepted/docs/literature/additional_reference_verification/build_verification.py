"""Six citation/source bundles, authenticated against retained source metadata.

Reads NetCDF attributes, text headers/footers and provenance JSON only. It does
not read benchmark scores, evaluation truth, model files or numerical arrays.
"""
from pathlib import Path
import datetime, hashlib, json, re, shutil
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def write(name,obj):
    (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,
                                    default=lambda value:value.tolist())+'\n',encoding='utf-8')

def paper(filename,url):
    m=read(ROOT/'raw'/filename)['message']
    return {'type':'journal_article','title':m['title'][0],
            'authors':[{'given':a.get('given'),'family':a['family']} for a in m['author']],
            'year':m['published']['date-parts'][0][0],
            'journal':m['container-title'][0],'volume':m.get('volume'),
            'issue':m.get('issue'),'pages':m.get('page'),'doi':m['DOI'],
            'publisher':m['publisher'],'primary_publisher_url':url,
            'publisher_deposited_metadata':filename,
            'published_print':m.get('published-print'),
            'published_online':m.get('published-online')}

def main():
    retrieval=read(ROOT/'retrieval_manifest.json')
    for item in retrieval:
        assert item['verification_status']=='RETRIEVED'
        assert sha(ROOT/item['filename'])==item['sha256']
    checks=[]
    amv_manifest=read(WORKSPACE/'work/amv_rebuild/retrieval_manifest.json')
    source=amv_manifest['sources'][0]
    nc_path=WORKSPACE/'work/amv_rebuild'/source['local_file']
    actual=sha(nc_path)
    assert actual==source['sha256']
    with Dataset(nc_path) as nc:
        keys=['id','title','product_version','dataset_citation_product',
              'dataset_citation_version','dataset_citation_institution','institution',
              'metadata_link','source','references','climatology','acknowledgment',
              'date_created','date_modified','time_coverage_start','time_coverage_end',
              'geospatial_lat_resolution','geospatial_lon_resolution']
        attrs={k:nc.getncattr(k) for k in keys if k in nc.ncattrs()}
        variable_attrs={v:{k:nc.variables[v].getncattr(k) for k in ['units','long_name','standard_name']
                           if k in nc.variables[v].ncattrs()} for v in ['sst','ssta']}
    assert attrs['product_version']=='v6' and variable_attrs['sst']['units']=='degree_C'
    checks.append({'source':'ERSSTv6','source_path':str(nc_path.relative_to(WORKSPACE)),
                   'sha256':actual,'manifest_sha256':sha(WORKSPACE/'work/amv_rebuild/retrieval_manifest.json'),
                   'url':source['url'],'retrieved_utc':source['retrieved_utc'],'status':'PASS'})
    power_path=WORKSPACE/'work/power_imd_rebuild/power_native_extraction_receipt.json'
    power=read(power_path)
    z_manifest=read(WORKSPACE/'work/power_imd_rebuild/zarr_metadata_manifest.json')
    power_metadata={}
    global_keys=['id','version','institution','project','references','time_standard',
                 'date_created','geospatial_lat_resolution','geospatial_lon_resolution',
                 'time_coverage_start','time_coverage_end','comment']
    for name,record in power['sources'].items():
        z_path=WORKSPACE/f'work/power_imd_rebuild/{name}_monthly_zmetadata.json'
        z=read(z_path)['metadata']
        manifest_record=next(x for x in z_manifest if x['source_root']==record['root_url'])
        assert sha(z_path)==manifest_record['sha256']
        assert z['.zattrs']==record['official_global_attributes']
        for var in record['requested_variables']:
            assert z[var+'/.zattrs']==record['official_variable_attributes'][var]
        power_metadata[name]={
            'root_url':record['root_url'],'global_attributes':{k:z['.zattrs'].get(k) for k in global_keys},
            'variable_attributes':{v:{k:z[v+'/.zattrs'].get(k) for k in ['units','long_name','cell_methods']}
                                   for v in record['requested_variables']},
            'zmetadata_sha256':sha(z_path),'retrieved_utc':manifest_record['retrieved_at_utc']}
        checks.append({'source':'POWER '+name,'source_path':str(z_path.relative_to(WORKSPACE)),
                       'sha256':sha(z_path),'receipt_sha256':sha(power_path),'status':'PASS'})
    mei_path=WORKSPACE/'work/drive/teleconnections/MEIv2_monthly_1979-present.txt'
    mei_hash=sha(mei_path)
    assert mei_hash=='90a376fa1b5a07b512ee072d10d4d16312f4f20cbc679979311376aa853dfb68'
    lines=mei_path.read_text(encoding='utf-8-sig').splitlines()
    mei_attrs={'source_path':str(mei_path.relative_to(WORKSPACE)),'sha256':mei_hash,
               'first_line':lines[0].strip(),
               'footer':[x.strip() for x in lines if x.strip().startswith(('Multivariate','https://','Row values'))],
               'missing_sentinel':-999.0,
               'jra_vintage_identifier_in_file':None}
    assert 'JRA55' not in '\n'.join(lines) and 'JRA3Q' not in '\n'.join(lines)
    checks.append({'source':'frozen MEIv2','source_path':mei_attrs['source_path'],
                   'sha256':mei_hash,'version_footer_present':'Version 2' in '\n'.join(lines),'status':'PASS'})
    gldas_path=WORKSPACE/'outputs/03_benchmark/caadapfl_benchmark/tools/derived/gldas_full/manifest.json'
    gldas=read(gldas_path)
    assert sha(gldas_path)=='049980766a7bb4c513dd0c86371573af7f55705831031989dc658bfe6add0042'
    assert gldas['status']=='PASS'
    gldas_attrs={k:gldas[k] for k in ['status','source_sha256','rows','variables','calendar',
                                    'unit_policy','units','limits']}
    gldas_attrs['manifest_sha256']=sha(gldas_path)
    checks.append({'source':'GLDAS supplement lineage','source_path':str(gldas_path.relative_to(WORKSPACE)),
                   'sha256':sha(gldas_path),'status':'PASS'})
    local_sources=[]
    copy_names=[('work/power_imd_rebuild/power_sources_doc.html','power_sources_frozen.html'),
                ('work/power_imd_rebuild/power_aws_doc.html','power_aws_frozen.html'),
                ('work/era5_rebuild/ee_review/source_evidence/NASA_README_GLDAS2.pdf','NASA_README_GLDAS2_frozen.pdf')]
    for relative,name in copy_names:
        source_path=WORKSPACE/relative
        assert source_path.exists(),str(source_path)
        expected={'power_sources_frozen.html':'68a23d1eb267dea6bc23ab74203ed0521ba5bfd9069a4eccdd959e04f8ce55ed',
                  'power_aws_frozen.html':'4419f0dff5f6bc4fdf57aaf247f290c97fe2cdbed76a6fbc97b738a83e128158'}.get(name)
        if expected:
            assert sha(source_path)==expected
        shutil.copyfile(source_path,ROOT/'raw'/name)
        local_sources.append({'source_path':str(source_path.relative_to(WORKSPACE)),
                             'retained_copy':'raw/'+name,'sha256':sha(source_path)})
    write('local_native_attributes.json',{'ersst6':{'global_attributes':attrs,'variable_attributes':variable_attrs},
                                          'power':power_metadata,'mei_v2':mei_attrs,'gldas21':gldas_attrs})
    write('local_source_authentication.json',{'status':'PASS','checks':checks,'retained_official_documents':local_sources,
          'numerical_arrays_read':False,'model_or_evaluation_files_read':False})
    b=paper('barnston_crossref.json','https://www.tandfonline.com/doi/abs/10.1080/07055900.1997.9649597')
    assert b['authors'][0]['family']=='Barnston' and b['year']==1997
    assert b['published_online']['date-parts'][0]==[2010,11,19]
    p1=paper('ersst6_part1_crossref.json','https://journals.ametsoc.org/view/journals/clim/38/4/JCLI-D-23-0707.1.xml')
    p2=paper('ersst6_part2_crossref.json','https://journals.ametsoc.org/view/journals/clim/38/4/JCLI-D-24-0185.1.xml')
    assert p1['year']==p2['year']==2025 and p1['authors'][1]['given']=='Xungang'
    z=paper('mei_v2_zhang_crossref.json','https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019GL083946')
    m=paper('merra2_gelaro_crossref.json','https://journals.ametsoc.org/view/journals/clim/30/14/jcli-d-16-0758.1.xml')
    d=read(ROOT/'raw/gldas21_datacite.json')['data']['attributes']
    assert d['publicationYear']==2020 and d['doi'].lower()=='10.5067/e7tyrxpjkwoq'
    cmr=read(ROOT/'raw/gldas21_cmr.json')['feed']['entry']
    assert len(cmr)==1 and cmr[0]['version_id']=='2.1'
    bundles=[
      {'id':'01_original_reference26','status':'VERIFIED_CORRECTION','publication':b,
       'original_anchor':'original manuscript reference26, paragraph P0851',
       'action':'Replace Bamston with Barnston; use citation year1997; retain2010 only as online-publication metadata.',
       'evidence_locators':['raw/barnston_crossref.json: message.author, published-print, published-online, volume, issue, page, DOI'],
       'access_limit':'Publisher landing was403 in direct browser opening; indexed publisher metadata and public publisher-deposited Crossref record verified without full-text access.'},
      {'id':'02_ersst_v6','status':'VERIFIED_VERSION_AND_PUBLICATIONS','organization':'NOAA/NESDIS/NCEI',
       'dataset_title':'NOAA monthly ERSSTv6 (in situ only)','dataset_version':'v6','dataset_doi':None,
       'official_product_url':'https://www.ncei.noaa.gov/products/extended-reconstructed-sst',
       'publications':[p1,p2],'frozen_snapshot_sha256':actual,'frozen_snapshot_retrieved_utc':source['retrieved_utc'],
       'native_grid':'2 degree latitude x 2 degree longitude','native_sst_units':'degree_C',
       'native_ssta_climatology':'1991-2020','derived_amv_climatology':'1951-1980 monthly climatology recomputed from absolute sst',
       'notes':['Product page has a placeholder pending a v6 data DOI; no v6 dataset DOI verified.',
                'Do not substitute ERSSTv5 article/data DOI for v6.',
                'The unsmoothed NorthAtlantic-minus-global AMV is researcher-derived; this does not identify it as an official NOAA AMO series.',
                'Archive snapshot is retrospective; no historical real-time release availability is claimed.'],
       'evidence_locators':['raw/ncei_ersst.html: Citation and Product Information','local_native_attributes.json: ersst6',
                            'raw/ersst6_part1_crossref.json: message','raw/ersst6_part2_crossref.json: message']},
      {'id':'03_mei_v2','status':'VERIFIED_PRODUCT_CITATION_VINTAGE_UNRESOLVED',
       'dataset_title':'Multivariate ENSO Index Version 2 (MEI.v2)','provider':'NOAA Physical Sciences Laboratory',
       'official_product_url':'https://psl.noaa.gov/enso/mei/','dataset_doi':None,'publication':z,
       'frozen_file_sha256':mei_hash,'frozen_reanalysis_vintage':None,
       'calendar_semantics':'12 overlapping two-month seasons, labeled by period-end month in this benchmark',
       'notes':['Provider lists Zhang etal2019 for multivariate monitoring; Wolter and Timlin2011 MEI.ext alone is insufficient as a dedicated MEIv2 dataset citation.',
                'Current page uses JRA3Q for its entire reconstructed time series, replacing JRA55 after March2024; it is not a JRA55/JRA3Q splice.',
                'Frozen input footer identifies Version2 and overlapping seasons, but not JRA55/JRA3Q. Do not retroactively assign the current page vintage to its bytes.',
                'Current overview1.5degree and legacy Methods2.5degree are different generations of documentation; avoid an unsupported archived-grid claim.'],
       'evidence_locators':['raw/noaa_mei_v2.html: Overview, Methods, References, Resources',
                            'raw/mei_v2_zhang_crossref.json: message','local_native_attributes.json: mei_v2']},
      {'id':'04_gldas_noah_3h_v21','status':'VERIFIED_CURRENT_DATASET_CITATION',
       'dataset_title':d['titles'][0]['title'],'creators':d['creators'],'publisher':d['publisher'],
       'publication_year':d['publicationYear'],'doi':'10.5067/E7TYRXPJKWOQ','official_product_url':d['url'],
       'short_name':'GLDAS_NOAH025_3H','version':'2.1','model_version':'Noah3.6',
       'citation_year_discrepancy':{'current_NASA_catalog_and_DataCite':2020,'older_README_example_page12':2016,
                                  'resolution':'Use2020 recommended current dataset citation; preserve older README discrepancy as evidence.'},
       'notes':['This DOI is for the3hourly0.25degreeV2.1 final product, not monthly, Early Product, GLDAS1 or another land-surface model.',
                'Retain the existing Rodell etal2004 GLDAS methods reference as well as this dataset reference.',
                'Published backward3h tavg/acc physical semantics underpin aggregation; raw time_bnds and custom GEE hour-property origin remain unverified.'],
       'evidence_locators':['raw/gldas21_datacite.json: data.attributes.creators,titles,publicationYear,doi,url',
                            'raw/gldas21_cmr.json: feed.entry[0].short_name,version_id,time_start',
                            'raw/NASA_README_GLDAS2_frozen.pdf: pages11-12 Table2 and section1.3; page23 Table4',
                            'local_native_attributes.json: gldas21']},
      {'id':'05_power_merra2_meteorology','status':'VERIFIED_FROZEN_SOURCE_AND_ACKNOWLEDGMENT',
       'provider':'NASA Langley Research Center, Prediction Of Worldwide Energy Resources (POWER)',
       'service':'native AWS analysis-ready monthly temporal Local Solar Time Zarr','version':'10.0.0',
       'access_date':'2026-10-07','upstream':'NASA Goddard Global Modeling and Assimilation Office, MERRA-2',
       'publication':m,'source_root':power_metadata['merra2']['root_url'],
       'native_grid':'0.5degree latitude x0.625degree longitude',
       'variables':power_metadata['merra2']['variable_attributes'],
       'acknowledgment_notes':['Credit NASA POWER/NASA LaRC and Earth Science Division funding; NASA/GMAO for MERRA-2.',
                               'Include actual monthly service/native version10.0.0/access date, adapting the generic POWER referencing example.',
                               'Named merra2 archive is used; do not describe current GEOS-IT near-real-time tail as an input here.',
                               'PRECTOTCORR stays monthly-mean mm/day; it is not a monthly rainfall total.'],
       'evidence_locators':['local_native_attributes.json: power.merra2','raw/power_referencing.html: Referencing Guide',
                            'raw/power_acknowledgements.html: Acknowledgement',
                            'raw/power_sources_frozen.html: Meteorology',
                            'raw/nasa_gmao_merra2.html: Overview Paper','raw/merra2_gelaro_crossref.json: message']},
      {'id':'06_power_srb_syn_solar','status':'VERIFIED_FROZEN_SOURCE_AND_ACKNOWLEDGMENT',
       'provider':'NASA POWER/NASA Langley Research Center','service':'native AWS monthly temporal LST Zarr',
       'version':'10.0.0','access_date':'2026-10-07','native_grid':'1degree x1degree',
       'source_roots':{k:power_metadata[k]['root_url'] for k in ['srb','syn1deg']},
       'study_source_calendar':{'SRB':'1984-01 through2000-12','CERES_SYN1deg':'2001-01 through2024-12'},
       'native_units':'W m-2','benchmark_units':'MJ m-2 day-1','conversion':'multiply monthly-mean W m-2 by86400/1000000 =0.0864',
       'upstream_release_edition_claim':'Current POWER documentation describes SRB4.0IP and SYN1degEdition4.2; frozen POWER native metadata pins version10.0.0/source families, not those upstream editions independently.',
       'dataset_doi':None,
       'acknowledgment_notes':['Credit NASA POWER/NASA LaRC; NASA/GEWEX SRB and NASA CERES SYN1deg source projects; NASA LaRC ASDC.',
                              'Provider documentation warns that multiple solar sources limit long-term trend inference across transitions.',
                              'Do not claim FLASHFlux inputs for the frozen1984-2024 SRB/SYN supplement, or availability before1984.',
                              'No source-specific solar DOI or release year is invented; POWER version is not the CERES/SRB edition.'],
       'evidence_locators':['local_native_attributes.json: power.srb and power.syn1deg',
                            'raw/power_sources_frozen.html: Solar', 'raw/power_acknowledgements.html: Acknowledgement',
                            'raw/nasa_srb_acknowledgement.html: Data Usage Acknowledgement','raw/power_referencing.html: Referencing Guide']}
    ]
    assert len(bundles)==6
    write('verified_metadata.json',{'schema':'bounded_reference_verification_v1','status':'PASS_WITH_EXPLICIT_VINTAGE_LIMITS',
          'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'bundle_count':6,'items':bundles,
          'scope':'citation metadata/source acknowledgments only; no manuscript writing, new data substitution or model-result reads'})
    print('PASS',len(bundles),'bundles;',len(checks),'local metadata/hash checks;',len(retrieval),'retained public responses')

if __name__=='__main__':
    main()
