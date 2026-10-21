from django.shortcuts import render
import pandas as pd
import numpy as np
 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
 
from aplicativosML.models import Dados_Apontamentos
 
def executarRegressaoLogistica(request):
    campos_do_banco = ['cpf_cnpj', 'cidade', 'codigo_cartorio', 'valor_original', 'valor_enviado', 'status']
    dados = Dados_Apontamentos.objects.values(*campos_do_banco)
    df_base = pd.DataFrame.from_records(dados)
    df_base.columns = df_base.columns.str.strip()
 
    # Tratamento de status padrão
    df_base['status'] = df_base['status'].replace('RECEBIDO PELO CARTÓRIO', 'PROTESTADO')
    df_base['status'] = df_base['status'].replace('PAGO PELO DEVEDOR', 'REPASSADO AO CREDOR')
    df_base['status'] = df_base['status'].replace('RETIRADO', 'CANCELADO')
 
    # ---------------------------------------------
    # 1. REGRESSÃO LOGÍSTICA (Binário: Protestado vs Outros)
    # ---------------------------------------------
    df_rl = df_base.copy()
    y_rl = df_rl['status'].apply(lambda x: 1 if str(x).strip().upper() == 'PROTESTADO' else 0)
 
    for col in df_rl.select_dtypes(include=['object']).columns:
        if col != 'status':
            try:
                df_rl[col] = pd.to_datetime(df_rl[col])
            except Exception:
                df_rl[col] = LabelEncoder().fit_transform(df_rl[col].astype(str))
 
    X_rl = df_rl.drop('status', axis=1)
    X_train_rl, X_test_rl, y_train_rl, y_test_rl = train_test_split(
        X_rl, y_rl, test_size=0.20, random_state=42, stratify=y_rl
    )
 
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train_rl, y_train_rl)
 
    scaler = StandardScaler()
    X_res_scaled = scaler.fit_transform(X_res)
    X_test_scaled = scaler.transform(X_test_rl)
 
    model_rl = LogisticRegression(max_iter=1000, class_weight='balanced')
    model_rl.fit(X_res_scaled, y_res)
    y_pred_rl = model_rl.predict(X_test_scaled)
 
    acc_rl = round(accuracy_score(y_test_rl, y_pred_rl) * 100, 2)
    prec_rl = round(precision_score(y_test_rl, y_pred_rl, zero_division=0) * 100, 2)
    rec_rl = round(recall_score(y_test_rl, y_pred_rl, zero_division=0) * 100, 2)
 
    # ---------------------------------------------
    # 2. NAIVE BAYES (Binário comparativo)
    # ---------------------------------------------
    df_nb = df_base.copy()
    unicidade_cidade = df_nb['cidade'].unique()
    cidade_map = {cid: i for i, cid in enumerate(unicidade_cidade)}
    df_nb['cidade_numero'] = df_nb['cidade'].map(cidade_map)
 
    y_nb = df_nb['status'].apply(lambda x: 1 if str(x).strip().upper() == 'PROTESTADO' else 0)
    X_nb = df_nb[['cidade_numero', 'codigo_cartorio']].fillna(0)
 
    X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(
        X_nb, y_nb, test_size=0.20, random_state=42, stratify=y_nb
    )
 
    modelo_nb = MultinomialNB()
    modelo_nb.fit(X_train_nb, y_train_nb)
    y_pred_nb = modelo_nb.predict(X_test_nb)
 
    acc_nb = round(accuracy_score(y_test_nb, y_pred_nb) * 100, 2)
    prec_nb = round(precision_score(y_test_nb, y_pred_nb, zero_division=0) * 100, 2)
    rec_nb = round(recall_score(y_test_nb, y_pred_nb, zero_division=0) * 100, 2)
 
    contexto = {
        'acc_nb': acc_nb,
        'prec_nb': prec_nb,
        'rec_nb': rec_nb,
        'acc_rl': acc_rl,
        'prec_rl': prec_rl,
        'rec_rl': rec_rl,
        'leader_acc': "Reg. Logística lidera" if acc_rl >= acc_nb else "Naive Bayes lidera",
        'leader_prec': "Reg. Logística lidera" if prec_rl >= prec_nb else "Naive Bayes lidera",
        'leader_rec': "Reg. Logística lidera" if rec_rl >= rec_nb else "Naive Bayes lidera",
    }
 
    return render(request, "resultado.html", contexto)