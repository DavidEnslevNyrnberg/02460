import pandas

it=pandas.read_sas('PUF_SAS_COMBINED_CMB_STU_QQQ/cy6_ms_cmb_stu_qqq.sas7bdat',format='sas7bdat', index=None, encoding=None, chunksize=1, iterator=True)

i=0
for line in it:
    while i<5:
        print(line)
        i=i+1
    break

print('hello')
