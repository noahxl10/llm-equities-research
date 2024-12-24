
from prediction_engine import ML

import dev_setup
dev_setup.set_environment_variables()
from db import db


ml = ML()

x = ml.get_feature_performances()
x = x.rename(columns={"Feature": "feature_id"})
x['feature_id'] = x['feature_id'].apply(str)
dbg = db.DB()
df = dbg.query_to_df('select feature_id, feature_question from dw.d_features')
df['feature_id'] = df['feature_id'].apply(str)
final_df = x.merge(df, how='left', on='feature_id')
ar = ml.ff_nn()
