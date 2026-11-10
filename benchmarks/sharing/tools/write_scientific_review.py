"""Bind reviewed estimates to Persian report and English reviewer addendum."""
from pathlib import Path
import json,sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import run_experiment as r

def main():
    assert r.load(ROOT/'qa/independent_numeric_qa.json')['status']=='PASS'
    assert r.load(ROOT/'qa/state_prediction_replay.json')['status']=='PASS'
    interpretation=r.load(ROOT/'reports/scientific_interpretation.json')
    assert interpretation['status']=='REVIEWED'
    summary=pd.read_csv(ROOT/'evaluation/principal_comparison.csv').set_index('model')
    cost=pd.read_csv(ROOT/'evaluation/fit_cost_40.csv').groupby('model').mean(numeric_only=True)
    ci=pd.read_csv(ROOT/'evaluation/paired_CI_7.csv');ci=ci[(ci.block_length_months==12)&(ci.metric=='macro_MSE_mm2')].set_index('contrast_id')
    tests=pd.read_csv(ROOT/'evaluation/annual_MSE_sign_tests.csv').set_index('contrast_id')
    labels={a['label']:a['label'] for a in r.load(ROOT/'protocol.json')['arms']}
    rows=[]
    for model in ['TRAIN_CLIMATOLOGY',*labels]:
        q=summary.loc[model];p=0 if model=='TRAIN_CLIMATOLOGY' else float(cost.loc[model,'logical_required_payload_bytes'])
        rows.append(dict(model=model,r2=float(q.R2_mean),rmse_mm=float(q.RMSE_mm_mean),msess=float(q.MSESS_mean),MSE_mm2=float(q.MSE_mm2_mean),payload_MiB=p/1024**2))
    effects=[]
    for identity,q in ci.iterrows():
        t=tests.loc[identity];effects.append(dict(id=identity,title=q.title,delta_MSE_mm2=float(q.delta),CI_low=float(q.CI95_low),CI_high=float(q.CI95_high),
            holm6=None if pd.isna(t.holm_p_6_primary) else float(t.holm_p_6_primary),holm7=float(t.holm_p_7_sensitivity)))
    costs=[dict(model=m,payload_MiB=float(q.logical_required_payload_bytes/1024**2),wall_s=float(q.train_wall_seconds),cpu_s=float(q.train_cpu_seconds),
                VAL_audit_bytes=int(q.all_round_VAL_audit_bytes),VAL_selection_bytes=int(q.three_checkpoint_VAL_selection_bytes_all_identities)) for m,q in cost.iterrows()]
    method_fa='۴۰ آموزش واقعی تازه: هشت سیاست × پنج seed جفت‌شده. شبکهٔ FULL12 با ۱۷٬۳۳۰ پارامتر، ۶۰ tick × سه epoch، مقدار به‌روزرسانی و جریان تصادفی محلی ثابت. آمار ورودی فقط از TRAIN ساخته می‌شود. خروجی اصلی tick60 است؛ همهٔ مدل‌های فدرال از state جهانیِ نهایی استفاده می‌کنند. انتخاب بهترین macro R² در VAL میان tickهای ۲۰/۴۰/۶۰ یک تحلیل ثانویه است. مقایسهٔ متمرکز همان مشاهده‌ها، ولی ۸۴۶۰ گام در برابر ۹۰۰۰ گام توزیع‌شده دارد؛ مقایسهٔ زمینه‌ای است.'
    limits_fa='دورهٔ ۲۰۱۷–۲۰۲۴ قبلاً دیده شده؛ این امتداد اکتشافی است. پنج منطقه هم‌پوشان‌اند و همهٔ اجراها روی یک میزبان CPU انجام شده‌اند. ۴۸۰ نتیجهٔ واقعیِ client-month و هشت بلوک سالانه داریم؛ seedها مشاهدهٔ بارش تازه نیستند. استانداردسازی محلی سازگاری مختصات پارامتر برای averaging را نیز تغییر می‌دهد. اشتراک آمار، تضمین حریم خصوصی نیست؛ هزینهٔ منطقیِ پیام با WAN، انرژی یا هزینهٔ پولی یکسان نیست. اطلاعات انتشار تاریخی و منفعت عملیاتی اندازه‌گیری نشده‌اند.'
    stats_fa='MSE اولیه برابر میانگین مساوی client و seed است؛ RMSE، R² و MSESS هرکدام جداگانه در همان سطح میانگین می‌شوند. CIها بازنمونه‌گیری مشترکِ تاریخ و seed با بلوک‌های ۱۲ ماهه‌اند؛ حساسیت ۶/۲۴ ماهه نیز محفوظ است. CIهای ۹۵٪ تعدیل چندگانه ندارند. شش پرسش اشتراک، خانوادهٔ Holm6 اصلاح‌شده پیش از ارزیابی هستند؛ S07 دربارهٔ proximal، ثانویه است. طرح اولیهٔ هفت‌مقایسه‌ای و Holm7 به‌عنوان حساسیت حفظ شده‌اند. اصلاح پس از شروع آموزش و پیش از خواندن EVAL ثبت شد. با هشت سال، کمینهٔ p دوسویه 0.0078125 است و Holm7 هرگز به 0.05 نمی‌رسد؛ عدم رد، اثبات برابری نیست.'
    cost_fa='پیام منطقی بدون فشرده‌سازی، برای همهٔ فرستنده/گیرنده‌ها، دریافت اولیه و تحویل نهایی شمرده شده است. اشتراک آمار TRAIN برابر ۶۹۲۰ بایت است. مدل float32 برابر ۶۹۳۲۰ بایت دارد. pooling از نمایش مهندسی‌شدهٔ float64 نمونه‌های TRAIN قیمت‌گذاری شده، که پنجره‌های هم‌پوشان آن اطلاعات تکراری دارند؛ حداقل حجم دادهٔ منبع نیست. هزینهٔ audit همهٔ tickها و انتخاب سه checkpoint جدا از آموزش ثبت شده؛ انتخاب یک هویت ۲۸۰ بایت و جهانی+شخصی ۵۶۰ بایت دارد. زمان آموزش/aggregation از audit و serialization جداست؛ این مقایسهٔ CPU محلی از نوسان بار میزبان مصون نیست.'
    sections=[dict(id='sharing-main',title='آزمایش تازه: دقت و مقدار اشتراک اطلاعات',query='sharing_primary',table='sharing_primary',body=interpretation['finding_fa']+'\n\n'+method_fa),
        dict(id='sharing-effects',title='اثرهای جفت‌شده و عدم‌قطعیت',query='sharing_effects',table='sharing_effects',body=interpretation['effects_fa']+'\n\n'+stats_fa),
        dict(id='sharing-cost',title='هزینهٔ آموزش، تبادل و انتخاب',query='sharing_cost',table='sharing_cost',body=interpretation['cost_fa']+'\n\n'+cost_fa),
        dict(id='sharing-contribution',title='سهم علمی افزوده و پاسخ داوران',query='sharing_primary',body=interpretation['contribution_fa']+'\n\n'+interpretation['reviewers_fa']+'\n\n'+limits_fa)]
    r.save(ROOT/'reports/report_content.json',dict(title='اشتراک اطلاعات و هزینه: آزمایش هدفمندِ ریویزن Systems',summary=interpretation['finding_fa'],sections=sections,
        queries=dict(sharing_primary=rows,sharing_effects=effects,sharing_cost=costs)))
    lines=['گزارش بازبینی آزمایش اشتراک اطلاعات — ۲۰۲۶-۱۰-۱۰',interpretation['finding_fa'],method_fa,'نتایج اصلی (fixed60):']
    lines += [f"{q['model']}: R2={q['r2']:.6f}; RMSE={q['rmse_mm']:.6f} mm; MSESS={q['msess']:.6f}; payload={q['payload_MiB']:.6f} MiB" for q in rows]
    lines += [interpretation['effects_fa'],stats_fa,'مقایسه‌های MSE با CI بلوک ۱۲ ماه:']
    lines += [f"{q['id']} {q['title']}: delta={q['delta_MSE_mm2']:.6f}; CI=[{q['CI_low']:.6f},{q['CI_high']:.6f}]; Holm6={q['holm6']}; Holm7={q['holm7']}" for q in effects]
    lines += [interpretation['cost_fa'],cost_fa,interpretation['contribution_fa'],interpretation['reviewers_fa'],limits_fa,
        'بازبینی: QA محاسبات مستقل همهٔ drawها/CIها و دفتر هزینه را دوباره محاسبه کرد. بازپخش همهٔ ۶۲۴۰۰ پیش‌بینی عصبی از stateهای محفوظ با اختلاف دقیقاً صفر انجام شد. کد آموزش و هویت اجرا قفل هستند. خطاهای احراز job، dtype هزینه، هزینهٔ انتخاب VAL و guard ارزیابی/بازپخش پیش از پذیرش نتایج اصلاح شدند. اسناد Word مرحلهٔ قبل در این آزمایش بازنویسی نشده‌اند؛ متن افزودهٔ پاسخ‌نامه همراه بسته است.']
    (ROOT/'reports/Scientific_Review_FA.txt').write_text('\n\n'.join(lines),encoding='utf-8')
    en=interpretation['reviewer_addendum_en']
    table='\n'.join(f"{q['model']}: macro RMSE {q['rmse_mm']:.4f} mm; macro R2 {q['r2']:.6f}; macro MSESS {q['msess']:.6f}; required payload {q['payload_MiB']:.6f} MiB." for q in rows)
    (ROOT/'reports/Reviewer_Addendum_Information_Sharing.txt').write_text(en+'\n\nFixed tick60 primary estimates:\n'+table+'\n\nThe original seven-test protocol, before-evaluation six-test amendment, Holm7 sensitivity, raw forecasts, saved states and independent QA accompany this addendum. This is draft text for integration into the manuscript and response; journal submission has not occurred.\n',encoding='utf-8')
    print('SCIENTIFIC_REVIEW_WRITTEN',flush=True)

if __name__=='__main__':main()
