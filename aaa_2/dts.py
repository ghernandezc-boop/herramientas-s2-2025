import json, boto3, pandas as pd
from io import BytesIO

s3 = boto3.client('s3')

def lambda_handler(event, context):
    bucket = event['Records'][0]['s3']['bucket']['name']
    key    = event['Records'][0]['s3']['object']['key']
    
    obj = s3.get_object(Bucket=bucket, Key=key)
    df  = pd.read_csv(BytesIO(obj['Body'].read()))
    
    # DTS rules
    df['Model']      = df['Model'].str.strip().map(MODEL_MAP)
    df['Year']       = (df['Year'] - 2010) / 15
    df['Mileage_KM'] = pd.to_numeric(df['Mileage_KM'].str.replace(' ', ''))
    df.fillna({'Engine_Size_L': df['Engine_Size_L'].median()}, inplace=True)
    
    out_buffer = BytesIO()
    df.to_parquet(out_buffer, index=False)
    out_buffer.seek(0)
    
    s3.put_object(Bucket='bmw-clean', Key=key.replace('.csv', '.parquet'), Body=out_buffer)
    return {'statusCode': 200, 'rows': len(df)}