#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
@Project ：agent_fingerprint
@File    ：XgbClf.py
@Description:

    XGBoost 三分类器 (XgbClfModel) —— human / script-bot / AI-agent

    ============================================================
    一、 整体架构: XGBoost 单模型 (无 Stacking)
    ============================================================
    模型层 (唯一一层): 1 个 XGBoost 多分类器 (multi:softprob)
                       超参由 Phase 2 逐步 GridSearchCV 取 Top-1 定出

    结构示意:
      57 维行为特征 X
          │
          └──→ XGBoost × 1 ──→ P(human) / P(bot) / P(agent)

    注: 本类按 utils/model/Xgb.py 的 Phase 框架逐段对齐 —— 除下列 3 处
        因「回归 → 三分类」必须改动外，落盘契约、输出契约、缓存机制全部同款:
          · objective  reg:squarederror → multi:softprob
          · 评估指标   R²/RMSE          → macro-F1 / per-class recall / 混淆矩阵
          · 切分方式   Purged K-Fold    → Stratified K-Fold
            （本课题样本是**地址截面**不是时序，不存在前视标签泄漏；
              但存在**地址级泄漏** —— 同一地址不能同时出现在训练与验证折，
              Phase 0 已按 address 去重保证）

    ============================================================
    二、 训练管线 (5 个 Phase, 训练顺序)
    ============================================================
    Phase 0 — 数据准备与校验 (标签/地址去重/辅助列/黑名单)
    Phase 1 — 4 步特征选择 (硬去重 → 软去重 → 重要性截断 → Top-N)
    Phase 2 — 逐步 GridSearchCV 自动调参 (XGBoost Top-1 超参)
    Phase 3 — Stratified K-Fold OOF 训练 (防泄漏的折外预测)
    Phase 4 — 全量 Retrain (沿用 Phase 2 超参) 落盘线上模型
    Phase 5 — ★ 二分类对照实验 (论文核心叙事，本课题特有)

    🔴 OOF 估计的适用边界（论文 5.2 节必须如实交代，不要写成"无偏"）:
      Phase 3 保证的是**折内无标签泄漏** —— 每个样本的概率都来自没见过它的
      折外模型。但 Phase 1 的特征选择与 Phase 2 的调参都是在**全量标签**上
      做的，两者都会给 Phase 3 的分数带来乐观偏差。严格无偏需要嵌套 CV
      （把 Phase 1+2 整体放进外层折内重跑）。
      本课题信号极强（behavior 集 0.98 量级），该偏差量级远小于结论余量，
      故保留现结构并在论文中声明，而不是改口径。

    ============================================================
    三、 Phase 0: 数据准备与校验 (训练前的「验票 + 清场」)
    ============================================================
    不训练任何模型; 任一检查不过直接 raise:
      1) 硬校验  : 标签列存在、取值在三类内、不与辅助列重名
      2) 地址去重: 同一 address 只保留一行 —— 防止地址级泄漏
      3) 辅助列抽出: address / 规模类特征只作标识，不进模型
      4) 黑名单剔除: exclude_field 整列移除

    ============================================================
    四、 Phase 5: 二分类对照实验（★ 论文核心叙事）
    ============================================================
    只用 human + bot 训一个二分类器（模拟已有工作，它们的标签体系里没有
    agent 这一类），再拿它去分 agent，统计 agent 被分到哪一类。

    对标 Web 域 arXiv 2607.26935: MLP 二分类漏检 39.1% / SAINT 漏检 34.5%

    🎯 无论本实验数字高于还是低于 Web 域，都是可写进论文的结论:
       · 高 → 链上问题更严重，二分类方法失效更彻底
       · 低 → 链上 agent 行为更接近 bot，需解释为什么
    这个「两头都不亏」的设计正是课题立论方式的实现。

    ============================================================
    五、 落盘契约 (与 Xgb.py 同款)
    ============================================================
      {save_path}/
        xgb_clf_full.pkl   — Phase 4 全量 retrain 模型（最终调用的模型）
        config.pkl         — feature_names / label_field / classes / 环境指纹
        metrics.json       — OOF 指标 + 混淆矩阵 + 二分类对照结果
        feature_report.csv — Phase 1 特征筛选全过程

    ============================================================
    六、 调用方式
    ============================================================
      # 训练
      M = XgbClfModel(df_train=df, save_path='data/model/base')
      M.train()

      # 预测（进程内缓存，首调自愈加载）
      proba = XgbClfModel.pred('data/model/base', X_pred)          # (N,3) 概率
      label = XgbClfModel.pred_label('data/model/base', X_pred)    # (N,)  类名
'''

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from conf import config as cfg  # noqa: E402
from src.utils.common import log  # noqa: E402

warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
except ImportError:
    raise SystemExit('\n[缺依赖] 需要 xgboost：pip install xgboost\n')

import joblib  # noqa: E402
from sklearn.metrics import (classification_report, confusion_matrix,  # noqa: E402
                             f1_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402


DEFAULT_LABEL_FIELD = 'group'
DEFAULT_ID_COL = 'address'
# 规模类：不进模型，避免用「活动多不多」作弊
DEFAULT_EXCLUDE = ['n_acts', 'span_days_raw', 'acts_per_day']
CLASSES = ['human', 'bot', 'agent']


# ================================================================
#  XgbClfModel — XGBoost 三分类器
# ================================================================
class XgbClfModel(object):
    """
    XGBoost 三分类器 (human / bot / agent)

    核心函数:
      train()                 Phase 0-5 全流程训练
      pred(save_path, X)      预测概率 (N,3)
      pred_label(save_path,X) 预测类名 (N,)
      warmup(save_path)       预热缓存

    Args:
        df_train      : pd.DataFrame, 含 label_field 与 id_col
        save_path     : str/Path, 落盘目录
        label_field   : str, 标签列名（默认 'group'）
        id_col        : str, 地址标识列（用于去重防泄漏，不进模型）
        exclude_field : list[str], 特征黑名单
        n_folds       : int, OOF 折数
        grid_cv       : int, GridSearchCV 内部折数
        top_n         : int, Phase 1 保留的最重要特征数
        corr_thres    : float, Phase 1 去重相关系数阈值
        min_keep      : int, 特征数下限（防御性）
        verbose       : bool
    """

    _CACHE = {}     # 进程内缓存 {save_path: (config, model)}

    def __init__(self, df_train=None, save_path=None,
                 label_field: str = DEFAULT_LABEL_FIELD,
                 id_col: str = DEFAULT_ID_COL,
                 exclude_field: list = None,
                 n_folds: int = 5, grid_cv: int = 3,
                 top_n: int = 40, corr_thres: float = 0.95,
                 min_keep: int = 10, verbose: bool = True):
        """
        初始化: 保存训练参数 + 落盘根目录

        Args:
            见类文档
        Returns:
            None
        """
        self.df_train = df_train
        self.save_path = Path(save_path) if save_path else None
        self.label_field = label_field
        self.id_col = id_col
        if exclude_field is not None and not isinstance(exclude_field, list):
            raise TypeError('exclude_field 只接受 list[str] 或 None')
        self.exclude_field = list(dict.fromkeys(
            (exclude_field or []) + DEFAULT_EXCLUDE))
        self.n_folds = n_folds
        self.grid_cv = grid_cv
        self.top_n = top_n
        self.corr_thres = corr_thres
        self.min_keep = min_keep
        self.verbose = verbose

        self.classes_ = None
        self.feature_names = None
        self.best_params = None
        self.tuning_log = []          # 逐轮调参过程（论文表 5.1 数据源）
        self.metrics = {}

    # ============================================================
    #  内部工具
    # ============================================================
    def _log(self, msg):
        if self.verbose:
            log(msg, 'info')

    def _warn(self, msg):
        log(msg, 'warn')

    @staticmethod
    def _artifact_names():
        return ['xgb_clf_full.pkl', 'config.pkl', 'metrics.json',
                'feature_report.csv']

    def _model_dir(self) -> Path:
        self.save_path.mkdir(parents=True, exist_ok=True)
        return self.save_path

    @staticmethod
    def _env_fingerprint() -> dict:
        """环境指纹 —— 预测端会比对，版本不符时告警。"""
        import sklearn
        return {'python': sys.version.split()[0],
                'xgboost': xgb.__version__,
                'sklearn': sklearn.__version__,
                'numpy': np.__version__,
                'pandas': pd.__version__}

    # ============================================================
    #  Phase 0 — 数据准备与校验
    # ============================================================
    def _phase0(self) -> tuple:
        """
        Phase 0: 时序无关的截面数据准备

        1) 硬校验  : 标签列存在、取值合法
        2) 地址去重: 🔴 同一 address 只保留一行 —— 防止地址级泄漏
                     （同一地址若同时进训练与验证折，等于把答案抄给模型）
        3) 辅助列抽出 + 黑名单剔除

        Returns:
            (X: pd.DataFrame, y: np.ndarray, feat_cols: list)
        """
        log('=' * 62, 'info')
        log('Phase 0 — 数据准备与校验', 'step')
        log('=' * 62, 'info')
        df = self.df_train

        # 1) 硬校验
        if df is None or df.empty:
            raise ValueError('df_train 为空')
        if self.label_field not in df.columns:
            raise ValueError(f'缺标签列 {self.label_field}')
        if self.label_field in self.exclude_field:
            raise ValueError(f'标签列 {self.label_field} 不能出现在 exclude_field')
        bad = set(df[self.label_field].unique()) - set(CLASSES)
        if bad:
            raise ValueError(f'标签出现非法取值 {bad}，合法值为 {CLASSES}')

        # 2) 地址去重（防地址级泄漏）
        n0 = len(df)
        if self.id_col in df.columns:
            df = df.drop_duplicates(subset=[self.id_col], keep='first')
            if len(df) < n0:
                self._warn(f'  按 {self.id_col} 去重: {n0} → {len(df)}'
                           f'（防止同一地址跨折泄漏）')

        # 3) 特征列 = 全部列 - 标签 - 标识 - 黑名单
        drop = set([self.label_field, self.id_col] + self.exclude_field)
        feat_cols = [c for c in df.columns
                     if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
        if not feat_cols:
            raise ValueError('剔除黑名单后没有剩余特征列')

        X = df[feat_cols].replace([np.inf, -np.inf], np.nan)
        y = df[self.label_field].values
        self.classes_ = [c for c in CLASSES if c in set(y)]

        vc = pd.Series(y).value_counts()
        self._log(f'  样本 {len(df)} 行 × {len(feat_cols)} 特征')
        self._log(f'  类别分布: {dict(vc)}')
        if len(self.classes_) < 2:
            raise ValueError(f'只有 {len(self.classes_)} 个类别，无法分类')
        if len(self.classes_) < 3:
            self._warn(f'  仅 {len(self.classes_)} 类，三分类退化为二分类')
        if vc.min() < self.n_folds:
            self.n_folds = max(2, int(vc.min()))
            self._warn(f'  最小类仅 {vc.min()} 个样本，折数下调为 {self.n_folds}')
        return X, y, feat_cols

    # ============================================================
    #  Phase 1 — 特征选择
    # ============================================================
    def _phase1_feature_select(self, X: pd.DataFrame, y: np.ndarray) -> list:
        """
        Phase 1: 4 步特征选择

          Step 0 硬去重  : |Spearman| > corr_thres 的特征对，保留重要性更大者
          Step 1 常数列  : 方差为 0 或全 NaN 的列直接剔除
          Step 2 重要性  : XGBoost gain 重要性为 0 的剔除
          Step 3 Top-N   : 按重要性截断

        Args:
            X / y
        Returns:
            list: 保留的特征名
        """
        log('=' * 62, 'info')
        log('Phase 1 — 特征选择（4 步）', 'step')
        log('=' * 62, 'info')
        report = []
        cols = list(X.columns)

        # Step 1（先做，成本最低）: 常数列 / 全 NaN 列
        drop_const = [c for c in cols
                      if X[c].isna().all() or X[c].nunique(dropna=True) <= 1]
        cols = [c for c in cols if c not in drop_const]
        for c in drop_const:
            report.append({'feature': c, 'step': 'const_or_allnan',
                           'kept': 0, 'importance': 0})
        self._log(f'  Step 1 常数/全NaN: 剔除 {len(drop_const)} 个 → 剩 {len(cols)}')

        # 先算一次重要性，供 Step 0/2/3 使用
        imp = self._quick_importance(X[cols], y)

        # Step 0 硬去重
        # 🔴 2026-09-13 修正：内层循环必须在「参照特征 a 自己出局」时立刻 break。
        #    原实现只在外层循环顶部判断 `a in drop_corr`，一旦 a 在内层成为
        #    loser，后续仍会拿**已被剔除的 a** 去和剩下的 b 比较：
        #      · a 再次落败 ⇒ 同一特征被重复写进 feature_report.csv
        #        （实测 base_model 报告 58 行 / 57 特征，f_method_entropy 重两次）
        #      · a 侥幸胜出 ⇒ 把 b 剔掉，而「留下的」a 其实早已出局 ⇒ 过度剔除
        #    实测本数据集上修正前后七套特征集的入选结果完全一致（无指标变化），
        #    但这是真 bug，换一份数据就会咬人。
        drop_corr = set()
        if len(cols) > 1:
            corr = X[cols].corr(method='spearman').abs()
            for i, a in enumerate(cols):
                if a in drop_corr:
                    continue
                for b in cols[i + 1:]:
                    if b in drop_corr:
                        continue
                    v = corr.loc[a, b]
                    if pd.notna(v) and v > self.corr_thres:
                        loser = b if imp.get(a, 0) >= imp.get(b, 0) else a
                        drop_corr.add(loser)
                        report.append({'feature': loser, 'step': 'corr_dedup',
                                       'kept': 0, 'importance': imp.get(loser, 0)})
                        if loser == a:
                            break       # a 已出局，不得再用它淘汰别人
        cols = [c for c in cols if c not in drop_corr]
        self._log(f'  Step 0 相关去重(|r|>{self.corr_thres}): '
                  f'剔除 {len(drop_corr)} 个 → 剩 {len(cols)}')

        # Step 2 重要性为 0
        drop_zero = [c for c in cols if imp.get(c, 0) <= 0]
        if len(cols) - len(drop_zero) >= self.min_keep:
            cols = [c for c in cols if c not in drop_zero]
            for c in drop_zero:
                report.append({'feature': c, 'step': 'zero_importance',
                               'kept': 0, 'importance': 0})
            self._log(f'  Step 2 零重要性: 剔除 {len(drop_zero)} 个 → 剩 {len(cols)}')
        else:
            self._log(f'  Step 2 跳过（剔除后会低于 min_keep={self.min_keep}）')

        # Step 3 Top-N
        if self.top_n and len(cols) > self.top_n:
            cols = sorted(cols, key=lambda c: -imp.get(c, 0))[:self.top_n]
            self._log(f'  Step 3 Top-{self.top_n}: → 剩 {len(cols)}')

        for c in cols:
            report.append({'feature': c, 'step': 'kept',
                           'kept': 1, 'importance': imp.get(c, 0)})
        if self.save_path:
            pd.DataFrame(report).sort_values(
                ['kept', 'importance'], ascending=False).to_csv(
                self._model_dir() / 'feature_report.csv', index=False)

        log(f'  ✅ 最终保留 {len(cols)} 个特征', 'ok')
        return cols

    def _quick_importance(self, X: pd.DataFrame, y: np.ndarray) -> dict:
        """用一棵浅 XGBoost 快速估计特征重要性（gain）。"""
        yc = pd.Series(y).map({c: i for i, c in enumerate(self.classes_)}).values
        m = xgb.XGBClassifier(
            n_estimators=120, max_depth=4, learning_rate=0.2,
            objective='multi:softprob', num_class=len(self.classes_),
            random_state=cfg.RANDOM_SEED, n_jobs=-1,
            tree_method='hist', eval_metric='mlogloss')
        m.fit(X.fillna(X.median(numeric_only=True)), yc)
        return dict(zip(X.columns, m.feature_importances_))

    # ============================================================
    #  Phase 2 — 逐步 GridSearchCV
    # ============================================================
    def _phase2_tune(self, X: pd.DataFrame, y: np.ndarray) -> dict:
        """
        Phase 2: 逐步 GridSearchCV 调参（分组扫描，避免组合爆炸）

        Args:
            X / y
        Returns:
            dict: Top-1 超参
        """
        log('=' * 62, 'info')
        log('Phase 2 — 逐步 GridSearchCV 调参', 'step')
        log('=' * 62, 'info')
        yc = pd.Series(y).map({c: i for i, c in enumerate(self.classes_)}).values
        Xf = X.fillna(X.median(numeric_only=True))

        params = {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.1,
                  'subsample': 0.9, 'colsample_bytree': 0.9,
                  'min_child_weight': 1, 'reg_lambda': 1.0}
        grids = [
            {'max_depth': [3, 4, 6], 'min_child_weight': [1, 3]},
            {'subsample': [0.7, 0.9, 1.0], 'colsample_bytree': [0.7, 0.9, 1.0]},
            {'learning_rate': [0.05, 0.1, 0.2], 'n_estimators': [200, 400]},
            {'reg_lambda': [0.5, 1.0, 3.0]},
        ]
        cv = StratifiedKFold(n_splits=min(self.grid_cv, int(pd.Series(yc).value_counts().min())),
                             shuffle=True, random_state=cfg.RANDOM_SEED)
        # 🔴 逐轮记录搜索过程 —— 论文表 5.1 的数据来源。
        #   只存最终 Top-1 超参是不够的: 审稿人要看的是「扫了哪些候选、
        #   每轮选了什么、分数抬升多少」，即搜索路径本身。
        self.tuning_log = []
        prev = None
        for i, g in enumerate(grids, 1):
            base = xgb.XGBClassifier(
                **params, objective='multi:softprob',
                num_class=len(self.classes_), random_state=cfg.RANDOM_SEED,
                n_jobs=-1, tree_method='hist', eval_metric='mlogloss')
            gs = GridSearchCV(base, g, cv=cv, scoring='f1_macro', n_jobs=-1)
            gs.fit(Xf, yc)
            params.update(gs.best_params_)
            self.tuning_log.append({
                'round': i,
                'group': ', '.join(g),
                'candidates': '; '.join(f'{k}={v}' for k, v in g.items()),
                'n_combos': int(len(gs.cv_results_['params'])),
                'chosen': '; '.join(f'{k}={v}' for k, v in gs.best_params_.items()),
                'cv_macro_f1': round(float(gs.best_score_), 4),
                'delta': (round(float(gs.best_score_) - prev, 4)
                          if prev is not None else None),
            })
            prev = float(gs.best_score_)
            self._log(f'  轮 {i}/{len(grids)} {list(g)} → '
                      f'{gs.best_params_}  macro-F1={gs.best_score_:.4f}')

        self.best_params = params
        log(f'  ✅ Top-1 超参: {params}', 'ok')
        return params

    # ============================================================
    #  Phase 3 — Stratified K-Fold OOF
    # ============================================================
    def _make_clf(self, params: dict):
        """按超参造一个 XGBoost 多分类器。"""
        return xgb.XGBClassifier(
            **params, objective='multi:softprob', num_class=len(self.classes_),
            random_state=cfg.RANDOM_SEED, n_jobs=-1,
            tree_method='hist', eval_metric='mlogloss')

    def _phase3_oof(self, X: pd.DataFrame, y: np.ndarray, params: dict) -> dict:
        """
        Phase 3: Stratified K-Fold OOF —— 防泄漏的折外预测

        🔴 本课题样本是**地址截面**不是时序，不存在前视标签泄漏，
           但存在**地址级泄漏** —— Phase 0 已按 address 去重保证。

        Args:
            X / y / params
        Returns:
            dict: OOF 指标 + 混淆矩阵
        """
        log('=' * 62, 'info')
        log(f'Phase 3 — Stratified {self.n_folds}-Fold OOF', 'step')
        log('=' * 62, 'info')
        cls2i = {c: i for i, c in enumerate(self.classes_)}
        yc = pd.Series(y).map(cls2i).values
        Xf = X.fillna(X.median(numeric_only=True))

        oof = np.zeros((len(X), len(self.classes_)))
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                              random_state=cfg.RANDOM_SEED)
        for k, (tr, va) in enumerate(skf.split(Xf, yc), 1):
            m = self._make_clf(params)
            # 🔴 2026-09-13 修正：不再把验证折当 eval_set 传入。
            #    本管线没有开 early_stopping_rounds，eval_set 只会记录指标、
            #    不产生 best_iteration（实测七套模型 config.pkl 的 n_estimators
            #    与 Phase 2 选定值完全一致 ⇒ 此前那条 best_iteration 分支是死代码）。
            #    但只要有人日后补上早停，落盘模型的树数就会由验证折决定 ——
            #    那是把验证信息带进最终模型的泄漏。直接拆掉，杜绝这条路径。
            m.fit(Xf.iloc[tr], yc[tr], verbose=False)
            oof[va] = m.predict_proba(Xf.iloc[va])
            self._log(f'  折 {k}/{self.n_folds} 完成')

        pred = np.array(self.classes_)[oof.argmax(axis=1)]
        mac = float(f1_score(y, pred, average='macro'))
        cm = confusion_matrix(y, pred, labels=self.classes_)
        rec = {c: float(recall_score(y, pred, labels=[c], average='macro',
                                     zero_division=0)) for c in self.classes_}

        log(f'  ✅ OOF macro-F1 = {mac:.4f}', 'ok')
        for c in self.classes_:
            log(f'     {c:<8} recall = {rec[c]:.4f}', 'info')
        print()
        print(classification_report(y, pred, labels=self.classes_,
                                    zero_division=0))

        return {'oof_macro_f1': mac, 'oof_recall': rec,
                'confusion_matrix': cm.tolist(), 'labels': self.classes_,
                'n_folds': self.n_folds}

    # ============================================================
    #  Phase 4 — 全量 Retrain 并落盘
    # ============================================================
    def _phase4_retrain(self, X: pd.DataFrame, y: np.ndarray,
                        params: dict, feat_cols: list):
        """
        Phase 4: 以 Phase 2 选定的超参在全量样本上重训 → 落盘线上模型

        🔴 树数一律沿用 Phase 2 的 n_estimators，不从 OOF 折反推。
           从验证折取 best_iteration 会把验证信息带进最终模型（见 Phase 3 注释）。

        Args:
            X / y / params / feat_cols
        Returns:
            模型对象
        """
        log('=' * 62, 'info')
        log('Phase 4 — 全量 Retrain 并落盘', 'step')
        log('=' * 62, 'info')
        p = dict(params)

        yc = pd.Series(y).map({c: i for i, c in enumerate(self.classes_)}).values
        med = X.median(numeric_only=True)
        model = self._make_clf(p)
        model.fit(X.fillna(med), yc, verbose=False)

        d = self._model_dir()
        joblib.dump(model, d / 'xgb_clf_full.pkl')
        joblib.dump({'feature_names': feat_cols,
                     'label_field': self.label_field,
                     'id_col': self.id_col,
                     'classes': self.classes_,
                     'impute_median': med.to_dict(),
                     'best_params': p,
                     'env': self._env_fingerprint(),
                     'trained_at': time.strftime('%Y-%m-%d %H:%M:%S')},
                    d / 'config.pkl')
        log(f'  ✅ 落盘 {d}', 'ok')
        for n in self._artifact_names():
            fp = d / n
            if fp.exists():
                log(f'     {n:<22} {fp.stat().st_size/1024:>8.1f} KB', 'info')
        return model

    # ============================================================
    #  Phase 5 — ★ 二分类对照实验（论文核心叙事）
    # ============================================================
    def _phase5_binary_control(self, X: pd.DataFrame, y: np.ndarray,
                               params: dict) -> dict:
        """
        Phase 5: ★ 二分类对照实验

        只用 human + bot 训一个二分类器（模拟已有工作），再拿它去分 agent，
        统计 agent 被分到哪一类。对标 Web 域 arXiv 2607.26935 的
        MLP 39.1% / SAINT 34.5% 漏检率。

        Args:
            X / y / params
        Returns:
            dict: 对照结果；样本不足时返回 {}
        """
        log('=' * 62, 'info')
        log('Phase 5 — ★ 二分类对照实验（论文核心叙事）', 'step')
        log('=' * 62, 'info')
        mask_tr = np.isin(y, ['human', 'bot'])
        mask_ag = y == 'agent'
        if mask_ag.sum() < 5 or len(set(y[mask_tr])) < 2:
            self._warn('  样本不足，跳过二分类对照')
            return {}

        Xf = X.fillna(X.median(numeric_only=True))
        p = {k: v for k, v in params.items()}
        m = xgb.XGBClassifier(
            **p, objective='binary:logistic', random_state=cfg.RANDOM_SEED,
            n_jobs=-1, tree_method='hist', eval_metric='logloss')
        yb = (y[mask_tr] == 'bot').astype(int)     # 1=bot, 0=human
        m.fit(Xf[mask_tr], yb, verbose=False)

        pr = m.predict(Xf[mask_ag])
        n = len(pr)
        as_bot = float((pr == 1).sum() / n * 100)
        as_human = 100.0 - as_bot

        log(f'  agent 样本 {n} 个，用 human/bot 二分类器判定：', 'ok')
        log(f'     → 判为 human : {as_human:.1f}%', 'info')
        log(f'     → 判为 bot   : {as_bot:.1f}%', 'info')
        log(f'  对标 Web 域 (arXiv 2607.26935): MLP 39.1% / SAINT 34.5% 漏检', 'info')
        return {'n_agent': n, 'as_human_pct': as_human, 'as_bot_pct': as_bot,
                'web_domain_mlp': 39.1, 'web_domain_saint': 34.5}

    # ============================================================
    #  训练入口
    # ============================================================
    def train(self) -> dict:
        """
        【训练入口】按 Phase 0 → 1 → 2 → 3 → 4 → 5 顺序训练

        Returns:
            dict: metrics（同时落盘 metrics.json）
        """
        t0 = time.time()
        X, y, _ = self._phase0()
        feat_cols = self._phase1_feature_select(X, y)
        X = X[feat_cols]
        params = self._phase2_tune(X, y)
        m3 = self._phase3_oof(X, y, params)
        self._phase4_retrain(X, y, params, feat_cols)
        m5 = self._phase5_binary_control(X, y, params)

        self.feature_names = feat_cols
        self.metrics = {**m3, 'binary_control': m5,
                        'n_samples': int(len(X)),
                        'n_features': len(feat_cols),
                        'best_params': params,
                        'tuning_rounds': self.tuning_log,
                        'elapsed_s': round(time.time() - t0, 1)}
        if self.save_path:
            (self._model_dir() / 'metrics.json').write_text(
                json.dumps(self.metrics, ensure_ascii=False, indent=2))
        log('=' * 62, 'info')
        log(f'训练完成，用时 {self.metrics["elapsed_s"]}s | '
            f'OOF macro-F1 = {m3["oof_macro_f1"]:.4f}', 'ok')
        log('=' * 62, 'info')
        return self.metrics

    # ============================================================
    #  预测入口
    # ============================================================
    @classmethod
    def _load_bundle(cls, save_path):
        """唯一加载入口: 首调自愈加载并缓存（每进程一次）。"""
        key = str(Path(save_path).resolve())
        if key in cls._CACHE:
            return cls._CACHE[key]
        d = Path(save_path)
        cfg_p, mdl_p = d / 'config.pkl', d / 'xgb_clf_full.pkl'
        if not cfg_p.exists() or not mdl_p.exists():
            raise FileNotFoundError(f'{save_path} 下缺 config.pkl / xgb_clf_full.pkl，'
                                    f'请先 train()')
        conf = joblib.load(cfg_p)
        model = joblib.load(mdl_p)
        # 环境指纹比对（不阻断，只告警）
        now = cls._env_fingerprint()
        diff = {k: (v, now.get(k)) for k, v in conf.get('env', {}).items()
                if now.get(k) != v}
        if diff:
            log(f'⚠️ 训练/预测环境不一致: {diff}', 'warn')
        cls._CACHE[key] = (conf, model)
        return conf, model

    @classmethod
    def warmup(cls, save_path):
        """预热缓存（长驻服务启动时调用）。"""
        cls._load_bundle(save_path)
        log(f'warmup 完成: {save_path}', 'ok')

    @classmethod
    def pred(cls, save_path, X_Pred: pd.DataFrame) -> np.ndarray:
        """
        【预测入口】输出三类概率

        Args:
            save_path: 训练时的落盘目录
            X_Pred:    至少含 config.feature_names 全部列；多余列自动过滤
        Returns:
            np.ndarray, shape=(N, 3)，列顺序 = config['classes']
        """
        conf, model = cls._load_bundle(save_path)
        names = conf['feature_names']
        missing = [c for c in names if c not in X_Pred.columns]
        if missing:
            raise ValueError(f'X_Pred 缺少训练阶段保留的特征: {missing[:5]}'
                             f'{"..." if len(missing) > 5 else ""}'
                             f'（共 {len(missing)} 个）')
        X = X_Pred[names].replace([np.inf, -np.inf], np.nan)
        X = X.fillna(pd.Series(conf['impute_median']))
        return model.predict_proba(X)

    @classmethod
    def pred_label(cls, save_path, X_Pred: pd.DataFrame) -> np.ndarray:
        """
        【预测入口】输出类名

        Args:
            save_path / X_Pred
        Returns:
            np.ndarray, shape=(N,)，取值 human / bot / agent
        """
        conf, _ = cls._load_bundle(save_path)
        proba = cls.pred(save_path, X_Pred)
        return np.array(conf['classes'])[proba.argmax(axis=1)]
