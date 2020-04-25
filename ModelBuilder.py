from fastai.text import *
from fastai.tabular import *
from pathlib import Path
import pandas as pd

rootdir = Path('./')

# Get the tweets here
df = pd.read_csv(rootdir/'tweets.csv')


# Language Model
data_lm = (TextList
           .from_df(df, cols="text")
           .split_by_rand_pct(valid_pct=0.2, seed=42)
           .label_for_lm()
           .databunch())

learn_lm = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)
learn_lm.fit_one_cycle(4, 1e-1)

learn_lm.save_encoder('lm_enc')


# Text Model
data_textm = (TextList
           .from_df(df, cols="text", vocab=data_lm.train_ds.vocab)
           .split_by_rand_pct(valid_pct=0.2, seed=42)
           .label_from_df(cols="annotation")
           .databunch())

learn_textm = text_classifier_learner(data_textm, AWD_LSTM, drop_mult=0.5)
learn_textm.load_encoder('lm_enc')
learn_textm.fit_one_cycle(4, 1e-2)

learn_textm.export(file = 'model_text.pkl')


# Tabular Model
dep_var = 'annotation'
cat_names = ['has_media', 'user_no_profile_image', 'user_verified']
cont_names = ['num_hashtags', 'num_likes', 'num_mentions', 'num_retweets', 'num_urls', 'user_num_favourite_tweets',
              'user_num_followers', 'user_num_friends', 'user_num_lists', 'user_num_tweets']
procs = [FillMissing, Categorify, Normalize]

data_tabular = (TabularList
           .from_df(df, cat_names=cat_names, cont_names=cont_names, procs=procs)
           .split_by_rand_pct(valid_pct=0.2, seed=42)
           .label_from_df(cols="annotation")
           .databunch())

learn_tabular = tabular_learner(data_tabular, layers=[200,100], metrics=accuracy)
learn_tabular.fit_one_cycle(1, 1e-2)
learn_tabular.export(file = 'model_tabular.pkl')


# Ensemble Model

pred_tensors_textm, pred_tensors_target = learn_textm.get_preds(DatasetType.Valid, ordered=True)
pred_tensors_tabular = learn_tabular.get_preds(DatasetType.Valid)[0]

preds_textm = pd.DataFrame(pred_tensors_textm.numpy())
preds_tabular = pd.DataFrame(pred_tensors_tabular.numpy())
preds_target = pd.DataFrame(pred_tensors_target.numpy())

ensemble_df = (preds_textm
               .join(preds_tabular, how='left', lsuffix='_textm', rsuffix='_tabular')
               .join(preds_target)
               .rename(columns = {0: "target" }))

dep_var = 'target'
cat_names = []
cont_names = ['0_textm', '0_tabular', '1_textm', '1_tabular']
procs = [FillMissing, Categorify, Normalize]

data_ensemble = (TabularList
           .from_df(ensemble_df, cat_names=cat_names, cont_names=cont_names, procs=procs)
           .split_by_rand_pct(valid_pct=0.2, seed=42)
           .label_from_df(cols="target")
           .databunch())

learn_ensemble = tabular_learner(data_ensemble, layers=[200,100], metrics=accuracy)
learn_ensemble.fit_one_cycle(2, 1e-2)
learn_ensemble.export(file = 'model_ensemble.pkl')