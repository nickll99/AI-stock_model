"""数据验证器"""
import pandas as pd
import numpy as np
from typing import List, Dict


class DataValidator:
    """数据验证器"""
    
    def check_missing_values(self, df: pd.DataFrame) -> bool:
        """
        检查数据是否有缺失值
        
        Args:
            df: 数据DataFrame
            
        Returns:
            True表示没有缺失值，False表示有缺失值
        """
        if df.empty:
            return False
        
        # 检查关键字段是否有缺失
        critical_columns = ['open', 'high', 'low', 'close', 'vol']
        for col in critical_columns:
            if col in df.columns and df[col].isna().any():
                return False
        
        return True
    
    def check_price_anomalies(self, df: pd.DataFrame, max_pct_change: float = 20.0) -> List[str]:
        """
        检测价格异常（如涨跌幅超过限制）
        
        Args:
            df: 数据DataFrame
            max_pct_change: 最大涨跌幅限制（默认20%，考虑ST股和新股）
            
        Returns:
            异常日期列表
        """
        anomalies = []
        
        if df.empty or 'pct_chg' not in df.columns:
            return anomalies
        
        # 检查涨跌幅异常
        for idx, row in df.iterrows():
            if pd.notna(row['pct_chg']) and abs(row['pct_chg']) > max_pct_change:
                # 排除涨跌停情况
                if 'limit_status' not in df.columns or pd.isna(row.get('limit_status')) or row.get('limit_status') == 0:
                    anomalies.append(str(idx))
        
        return anomalies
    
    def check_price_consistency(self, df: pd.DataFrame) -> List[str]:
        """
        检查价格一致性（high >= close >= low, high >= open >= low）
        
        Args:
            df: 数据DataFrame
            
        Returns:
            不一致的日期列表
        """
        inconsistencies = []
        
        if df.empty:
            return inconsistencies
        
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return inconsistencies
        
        for idx, row in df.iterrows():
            if pd.notna(row['high']) and pd.notna(row['low']):
                # 检查最高价 >= 最低价
                if row['high'] < row['low']:
                    inconsistencies.append(f"{idx}: high < low")
                    continue
                
                # 检查收盘价在合理范围内
                if pd.notna(row['close']):
                    if row['close'] > row['high'] or row['close'] < row['low']:
                        inconsistencies.append(f"{idx}: close out of range")
                
                # 检查开盘价在合理范围内
                if pd.notna(row['open']):
                    if row['open'] > row['high'] or row['open'] < row['low']:
                        inconsistencies.append(f"{idx}: open out of range")
        
        return inconsistencies
    
    def check_volume_anomalies(self, df: pd.DataFrame, threshold_multiplier: float = 10.0) -> List[str]:
        """
        检查成交量异常（成交量突然放大）
        
        Args:
            df: 数据DataFrame
            threshold_multiplier: 异常倍数阈值（默认10倍）
            
        Returns:
            异常日期列表
        """
        anomalies = []
        
        if df.empty or 'vol' not in df.columns:
            return anomalies
        
        # 计算成交量的移动平均
        df_copy = df.copy()
        df_copy['vol_ma20'] = df_copy['vol'].rolling(window=20, min_periods=1).mean()
        
        for idx, row in df_copy.iterrows():
            if pd.notna(row['vol']) and pd.notna(row['vol_ma20']) and row['vol_ma20'] > 0:
                if row['vol'] > row['vol_ma20'] * threshold_multiplier:
                    anomalies.append(str(idx))
        
        return anomalies
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        生成数据质量报告
        
        Args:
            df: 数据DataFrame
            
        Returns:
            数据质量报告字典
        """
        report = {
            'total_records': len(df),
            'date_range': {
                'start': str(df.index.min()) if not df.empty else None,
                'end': str(df.index.max()) if not df.empty else None
            },
            'missing_values': {},
            'price_anomalies': [],
            'price_inconsistencies': [],
            'volume_anomalies': [],
            'quality_score': 100.0
        }
        
        if df.empty:
            report['quality_score'] = 0.0
            return report
        
        # 检查缺失值
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                report['missing_values'][col] = {
                    'count': int(missing_count),
                    'percentage': float(missing_count / len(df) * 100)
                }
        
        # 检查各类异常
        report['price_anomalies'] = self.check_price_anomalies(df)
        report['price_inconsistencies'] = self.check_price_consistency(df)
        report['volume_anomalies'] = self.check_volume_anomalies(df)
        
        # 计算质量分数
        deductions = 0
        
        # 关键字段缺失扣分
        critical_cols = ['open', 'high', 'low', 'close', 'vol']
        for col in critical_cols:
            if col in report['missing_values']:
                deductions += report['missing_values'][col]['percentage'] * 0.5
        
        # 异常数据扣分
        total_anomalies = (
            len(report['price_anomalies']) + 
            len(report['price_inconsistencies']) + 
            len(report['volume_anomalies'])
        )
        if len(df) > 0:
            anomaly_rate = total_anomalies / len(df) * 100
            deductions += anomaly_rate * 0.3
        
        report['quality_score'] = max(0.0, 100.0 - deductions)
        
        return report
