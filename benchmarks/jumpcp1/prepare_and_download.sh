mkdir ./features/
mkdir ./features/cellprofiler

mkdir ./features/cellprofiler/BR00117010/
mkdir ./features/cellprofiler/BR00117011/
mkdir ./features/cellprofiler/BR00117012/
mkdir ./features/cellprofiler/BR00117013/
mkdir ./features/cellprofiler/BR00117024/
mkdir ./features/cellprofiler/BR00117025/
mkdir ./features/cellprofiler/BR00117026/


aws s3 cp s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/workspace/profiles/2020_11_04_CPJUMP1/BR00117010/BR00117010_normalized_feature_select_negcon_batch.csv.gz ./features/cellprofiler/BR00117010/BR00117010_normalized_feature_select_negcon_batch.csv.gz --no-sign-request
aws s3 cp s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/workspace/profiles/2020_11_04_CPJUMP1/BR00117011/BR00117011_normalized_feature_select_negcon_batch.csv.gz ./features/cellprofiler/BR00117011/BR00117011_normalized_feature_select_negcon_batch.csv.gz --no-sign-request
aws s3 cp s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/workspace/profiles/2020_11_04_CPJUMP1/BR00117012/BR00117012_normalized_feature_select_negcon_batch.csv.gz ./features/cellprofiler/BR00117012/BR00117012_normalized_feature_select_negcon_batch.csv.gz --no-sign-request
aws s3 cp s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/workspace/profiles/2020_11_04_CPJUMP1/BR00117013/BR00117013_normalized_feature_select_negcon_batch.csv.gz ./features/cellprofiler/BR00117013/BR00117013_normalized_feature_select_negcon_batch.csv.gz --no-sign-request
aws s3 cp s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/workspace/profiles/2020_11_04_CPJUMP1/BR00117024/BR00117024_normalized_feature_select_negcon_batch.csv.gz ./features/cellprofiler/BR00117024/BR00117024_normalized_feature_select_negcon_batch.csv.gz --no-sign-request
aws s3 cp s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/workspace/profiles/2020_11_04_CPJUMP1/BR00117025/BR00117025_normalized_feature_select_negcon_batch.csv.gz ./features/cellprofiler/BR00117025/BR00117025_normalized_feature_select_negcon_batch.csv.gz --no-sign-request
aws s3 cp s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/workspace/profiles/2020_11_04_CPJUMP1/BR00117026/BR00117026_normalized_feature_select_negcon_batch.csv.gz ./features/cellprofiler/BR00117026/BR00117026_normalized_feature_select_negcon_batch.csv.gz --no-sign-request