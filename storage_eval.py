#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyStorage-Eval: 电化学储能电站技术经济评价工具

严格遵循:
- DL/T 2919-2025《电化学储能电站经济评价导则》
- 《电化学储能项目边界条件取值表》

核心逻辑: 全生命周期现金流折现 (DCF)
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, Literal
import pandas as pd
import numpy as np
import numpy_financial as npf

# ==============================================================================
# 常量定义 (依据 DLT2919-2025 和边界条件取值表)
# ==============================================================================

class StorageConstants:
    """储能项目常量配置"""

    # ========== 收益模式 ==========
    MODE_ARBITRAGE = 'arbitrage'           # 峰谷套利
    MODE_CAPACITY = 'capacity'             # 容量租赁
    MODE_ANCILLARY = 'ancillary'           # 辅助服务
    MODE_HYBRID = 'hybrid'                 # 混合模式

    # ========== 税率 (依据 DLT2919-2025 4.2.14) ==========
    VAT_ELECTRICITY = 0.13                # 电力销售/辅助服务增值税 13%
    VAT_CAPACITY = 0.06                   # 容量租赁增值税 6%
    SURTAX_RATE = 0.10                    # 附加税率 10% (城建7%+教育3%)
    INCOME_TAX_RATE = 0.25                # 企业所得税率 25%

    # ========== 折旧相关 ==========
    DEPRECIATION_YEARS = 10                # 储能折旧年限 (边界表)
    DEPRECIATION_YEARS_NON_BATTERY = 15    # 非电池资产折旧年限 (DL/T 2919-2025 E.1.4)
    RESIDUAL_RATIO = 0.05                  # 残值率 5%
    DEPRECIATION_BASE_RATIO = 0.95         # 折旧基数比例
    BATTERY_ASSET_RATIO = 0.60             # 电池资产占比 (默认60%, 可配置)

    # ========== 项目期限 ==========
    CONSTRUCT_PERIOD = 1                   # 建设期 (年)
    OPERATION_PERIOD = 20                  # 运营期 (年)

    # ========== 运维费率 (依据边界表) ==========
    OM_FEE_PER_KW = 30.0                   # 按功率: 30元/kW/年
    OM_FEE_PER_KWH = 0.05                  # 按电量: 0.05元/kWh/年

    # ========== 电池更换策略 (依据 DLT2919-2025 E.1.8) ==========
    BATTERY_LIFE_PEAKING = 10             # 调峰项目电池寿命 10年
    BATTERY_LIFE_FREQUENCY = 4            # 调频项目电池寿命 4年

    # ========== 其他费用 ==========
    OTHER_COST_RATIO = 0.005               # 其他费用比例 0.5%

    # ========== 电池更换处理方式 ==========
    REPLACEMENT_EXPENSE = 'expense'        # 费用化 (当年一次性扣除)
    REPLACEMENT_CAPITALIZE = 'capitalize'  # 资本化 (计入固定资产)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ==============================================================================
# 异常定义
# ==============================================================================

class StorageProjectError(Exception):
    """储能项目评价基础异常"""
    pass


class InputValidationError(StorageProjectError):
    """输入参数验证失败异常"""
    pass


class CalculationError(StorageProjectError):
    """计算失败异常"""
    pass


# ==============================================================================
# 核心类: 储能项目
# ==============================================================================

class StorageProject:
    """
    电化学储能电站技经评价核心类

    严格遵循 DL/T 2919-2025《电化学储能电站经济评价导则》

    Attributes:
        power_mw: 装机功率 (MW)
        capacity_mwh: 额定容量 (MWh)
        efficiency: 系统效率 (Round-trip efficiency)
        static_invest: 静态投资 (万元)
        battery_life: 电池寿命 (年)
        replacement_mode: 电池更换处理方式
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        """
        初始化储能项目

        Args:
            params: 项目参数字典

            通用参数:
                - power_mw: 装机功率 (MW)
                - capacity_mwh: 额定容量 (MWh)
                - efficiency: 系统效率 (0-1)，如 0.85 表示 85%
                - static_invest: 静态投资 (万元)
                - hours_per_year: 年运行小时数，默认按充放电次数折算
                - loan_rate: 长期贷款利率，默认 0.049
                - capital_ratio: 资本金比例，默认 0.2
                - deductible_tax: 可抵扣进项税 (万元)，可选

            收益模式选择:
                - revenue_mode: 'arbitrage'(峰谷套利), 'capacity'(容量租赁),
                              'ancillary'(辅助服务), 'hybrid'(混合)

            峰谷套利模式 (revenue_mode='arbitrage'):
                - cycles_per_year: 年循环次数 (默认 330 天 × 1 次/天)
                - charge_price: 充电电价 (元/kWh)
                - discharge_price: 放电电价 (元/kWh)

            容量租赁模式 (revenue_mode='capacity'):
                - lease_capacity: 租赁容量 (MW)
                - lease_price: 租赁价格 (元/MW/年)

            辅助服务模式 (revenue_mode='ancillary'):
                - ancillary_type: 'peaking'(调峰) 或 'frequency'(调频)
                - ancillary_revenue: 年辅助服务收入 (万元)

            电池更换策略:
                - battery_life: 电池寿命 (年)，调频=4, 调峰=10
                - replacement_cost: 电池更换成本 (万元)，可选，默认按静态投资比例估算
                - replacement_mode: 'expense'(费用化) 或 'capitalize'(资本化)

        Raises:
            InputValidationError: 参数验证失败
        """
        self.p = params.copy()
        self._validate_and_init_params()
        self.df: Optional[pd.DataFrame] = None
        self.total_invest: float = 0.0
        self.const_interest: float = 0.0

    def _validate_and_init_params(self) -> None:
        """参数校验与标准化"""
        # 验证通用必需参数
        required_keys = ['power_mw', 'capacity_mwh', 'static_invest']
        missing_keys = [k for k in required_keys if k not in self.p]
        if missing_keys:
            raise InputValidationError(f"缺少必需参数: {missing_keys}")

        # 获取并验证通用参数
        self.power_mw = float(self.p.get('power_mw', 0))
        self.capacity_mwh = float(self.p.get('capacity_mwh', 0))
        self.efficiency = float(self.p.get('efficiency', 0.85))
        self.static_invest = float(self.p.get('static_invest', 0))
        self.loan_rate = float(self.p.get('loan_rate', 0.049))
        self.capital_ratio = float(self.p.get('capital_ratio', 0.2))

        # 数值范围验证
        if self.power_mw <= 0:
            raise InputValidationError("装机功率必须大于0")
        if self.capacity_mwh <= 0:
            raise InputValidationError("额定容量必须大于0")
        if not 0 < self.efficiency <= 1:
            raise InputValidationError("系统效率必须在 (0, 1] 范围内")
        if self.static_invest <= 0:
            raise InputValidationError("静态投资必须大于0")
        if not 0 < self.capital_ratio <= 1:
            raise InputValidationError("资本金比例必须在 (0, 1] 范围内")

        # 预计算贷款本金
        self.loan_principal = self.static_invest * (1 - self.capital_ratio)

        # 电池资产占比 (用于分离折旧计算)
        # 依据 DL/T 2919-2025 E.1.4: 电池部分按寿命折旧, 其余部分按15-20年折旧
        self.battery_asset_ratio = float(self.p.get('battery_asset_ratio', StorageConstants.BATTERY_ASSET_RATIO))
        if not 0 < self.battery_asset_ratio <= 1:
            raise InputValidationError("电池资产比例必须在 (0, 1] 范围内")

        # 获取收益模式
        self.revenue_mode = self.p.get('revenue_mode', StorageConstants.MODE_ARBITRAGE)

        # 验证特定模式参数
        if self.revenue_mode == StorageConstants.MODE_ARBITRAGE:
            # 峰谷套利模式参数
            self.cycles_per_year = int(self.p.get('cycles_per_year', 330))
            self.charge_price = float(self.p.get('charge_price', 0.3))
            self.discharge_price = float(self.p.get('discharge_price', 0.9))
            logger.info(f"模式: 峰谷套利, 循环{self.cycles_per_year}次/年, "
                       f"价差{self.discharge_price - self.charge_price:.2f}元/kWh")

        elif self.revenue_mode == StorageConstants.MODE_CAPACITY:
            # 容量租赁模式参数
            if 'lease_capacity' not in self.p or 'lease_price' not in self.p:
                raise InputValidationError("容量租赁模式需要参数: lease_capacity, lease_price")
            self.lease_capacity = float(self.p['lease_capacity'])
            self.lease_price = float(self.p['lease_price'])
            logger.info(f"模式: 容量租赁, 租赁{self.lease_capacity}MW, "
                       f"价格{self.lease_price}元/MW/年")

        elif self.revenue_mode == StorageConstants.MODE_ANCILLARY:
            # 辅助服务模式参数
            if 'ancillary_type' not in self.p or 'ancillary_revenue' not in self.p:
                raise InputValidationError("辅助服务模式需要参数: ancillary_type, ancillary_revenue")
            self.ancillary_type = self.p['ancillary_type']
            self.ancillary_revenue = float(self.p['ancillary_revenue'])

            # 设置电池寿命
            if self.ancillary_type == 'frequency':
                self.battery_life = StorageConstants.BATTERY_LIFE_FREQUENCY
            else:
                self.battery_life = StorageConstants.BATTERY_LIFE_PEAKING

            logger.info(f"模式: 辅助服务-{self.ancillary_type}, "
                       f"年收入{self.ancillary_revenue}万元, 电池寿命{self.battery_life}年")

        elif self.revenue_mode == StorageConstants.MODE_HYBRID:
            # 混合模式需要组合参数
            self.cycles_per_year = int(self.p.get('cycles_per_year', 330))
            self.charge_price = float(self.p.get('charge_price', 0.3))
            self.discharge_price = float(self.p.get('discharge_price', 0.9))
            self.ancillary_revenue = float(self.p.get('ancillary_revenue', 0))
            self.battery_life = int(self.p.get('battery_life', StorageConstants.BATTERY_LIFE_PEAKING))
            logger.info(f"模式: 混合模式")

        # 电池更换策略
        # 先设置默认值
        self.battery_life = int(self.p.get('battery_life', StorageConstants.BATTERY_LIFE_PEAKING))

        if 'battery_life' in self.p:
            self.battery_life = int(self.p['battery_life'])
        elif self.revenue_mode == StorageConstants.MODE_ANCILLARY:
            # 辅助服务模式根据类型设置电池寿命
            self.ancillary_type = self.p.get('ancillary_type', 'peaking')
            if self.ancillary_type == 'frequency':
                self.battery_life = StorageConstants.BATTERY_LIFE_FREQUENCY
            else:
                self.battery_life = StorageConstants.BATTERY_LIFE_PEAKING
        elif self.revenue_mode == StorageConstants.MODE_HYBRID:
            self.battery_life = int(self.p.get('battery_life', StorageConstants.BATTERY_LIFE_PEAKING))

        self.replacement_mode = self.p.get('replacement_mode', StorageConstants.REPLACEMENT_EXPENSE)

        # 电池更换成本（可选，默认按静态投资的70%估算）
        if 'replacement_cost' in self.p:
            self.replacement_cost = float(self.p['replacement_cost'])
        else:
            self.replacement_cost = self.static_invest * 0.7

        logger.info(f"项目参数验证通过: 功率={self.power_mw}MW, 容量={self.capacity_mwh}MWh, "
                   f"投资={self.static_invest}万元, 电池寿命={self.battery_life}年")

    def _calc_construction_interest(self) -> float:
        """
        计算建设期利息

        依据 DLT2919-2025 4.2.6
        简化逻辑: 假定资金年中均匀投入
        """
        interest = (self.loan_principal / 2) * self.loan_rate
        return interest

    def _get_om_rate(self, year_idx: int) -> float:
        """
        获取运维费率

        依据边界表: 锂电池储能按 30元/kW 或 0.05元/kWh 估算
        取两者中的较大值
        """
        om_by_power = self.power_mw * 1000 * StorageConstants.OM_FEE_PER_KW / 10000  # 万元
        # 估算年发电量用于计算按电量的运维费
        annual_discharge = self.capacity_mwh * self.cycles_per_year if hasattr(self, 'cycles_per_year') else self.capacity_mwh * 330
        # 修正: MWh → kWh → 元 → 万元的完整换算
        # annual_discharge (MWh) × 1000 (kWh/MWh) × 0.05 (元/kWh) / 10000 (元/万元)
        om_by_energy = annual_discharge * 1000 * StorageConstants.OM_FEE_PER_KWH / 10000  # 万元

        return max(om_by_power, om_by_energy)

    def calculate_cash_flow(self) -> pd.DataFrame:
        """
        核心引擎: 生成运营期现金流表

        依据 DLT2919-2025 第4章 财务分析方法

        Returns:
            包含完整现金流数据的DataFrame
        """
        try:
            # --- A. 建设期计算 ---
            const_interest = self._calc_construction_interest()

            # 流动资金 (按边界表，储能项目流动资金较小)
            working_capital = self.static_invest * 0.01

            # 动态总投资
            total_invest = self.static_invest + const_interest + working_capital

            # 增值税抵扣池初始化
            deductible_tax = self.p.get(
                'deductible_tax',
                self.static_invest / (1 + StorageConstants.VAT_ELECTRICITY) * StorageConstants.VAT_ELECTRICITY
            )

            # --- B. 初始化现金流表 ---
            years = np.arange(1, StorageConstants.OPERATION_PERIOD + 2)  # 1..21
            df = pd.DataFrame(index=years)

            cols = [
                'Charge_Cost', 'Discharge_Revenue', 'Lease_Revenue', 'Ancillary_Revenue',
                'Revenue_Inc', 'Revenue_Exc', 'Output_VAT', 'OM_Cost', 'VAT_Payable',
                'Surtax', 'Battery_Replacement', 'Depreciation', 'Profit_Total', 'Income_Tax',
                'Net_CF_Pre', 'Net_CF_After'
            ]
            for c in cols:
                df[c] = 0.0

            # 第1年 (建设期) 现金流出
            df.loc[1, 'Net_CF_Pre'] = -(self.static_invest + working_capital)
            df.loc[1, 'Net_CF_After'] = -(self.static_invest + working_capital)

            # --- C. 运营期逐年迭代 ---
            current_deductible = deductible_tax
            fixed_asset_value = self.static_invest + const_interest - deductible_tax

            # 分离电池资产与非电池资产折旧 (依据 DL/T 2919-2025 E.1.4)
            battery_asset_value = fixed_asset_value * self.battery_asset_ratio
            non_battery_asset_value = fixed_asset_value * (1 - self.battery_asset_ratio)

            # 电池资产按电池寿命折旧
            battery_depreciation_per_year = (
                battery_asset_value * StorageConstants.DEPRECIATION_BASE_RATIO / self.battery_life
            )
            # 非电池资产按15-20年折旧
            non_battery_depreciation_per_year = (
                non_battery_asset_value * StorageConstants.DEPRECIATION_BASE_RATIO / StorageConstants.DEPRECIATION_YEARS_NON_BATTERY
            )

            for y in range(2, StorageConstants.OPERATION_PERIOD + 2):
                op_year = y - 1

                # ========== 收入计算 ==========
                charge_cost = 0.0
                discharge_revenue = 0.0
                lease_revenue = 0.0
                ancillary_revenue = 0.0

                if self.revenue_mode == StorageConstants.MODE_ARBITRAGE or self.revenue_mode == StorageConstants.MODE_HYBRID:
                    # 峰谷套利收入 (依据 DLT2919-2025 公式 4.2.7-1)
                    # 年收入 = 循环次数 × (η × 容量 × 放电电价 - 容量 × 充电电价)
                    annual_revenue = self.cycles_per_year * (
                        self.efficiency * self.capacity_mwh * self.discharge_price * 10000
                        - self.capacity_mwh * self.charge_price * 10000
                    ) / 10000  # 万元

                    # 拆分为充电成本和放电收入（更清晰的现金流）
                    discharge_revenue = self.cycles_per_year * self.efficiency * self.capacity_mwh * self.discharge_price * 10000 / 10000  # 万元
                    charge_cost = self.cycles_per_year * self.capacity_mwh * self.charge_price * 10000 / 10000  # 万元

                    # 混合模式还要加上辅助服务收入
                    if self.revenue_mode == StorageConstants.MODE_HYBRID:
                        ancillary_revenue = self.ancillary_revenue

                elif self.revenue_mode == StorageConstants.MODE_CAPACITY:
                    # 容量租赁收入 (依据 DLT2919-2025 公式 4.2.8-1)
                    # 收入 = 租赁容量 × 租赁价格
                    lease_revenue = self.lease_capacity * self.lease_price

                elif self.revenue_mode == StorageConstants.MODE_ANCILLARY:
                    # 辅助服务收入
                    ancillary_revenue = self.ancillary_revenue

                # 计算含税和不含税收入
                if self.revenue_mode == StorageConstants.MODE_CAPACITY:
                    # 容量租赁增值税率 6%
                    rev_inc = lease_revenue
                    rev_exc = lease_revenue / (1 + StorageConstants.VAT_CAPACITY)
                    output_vat = rev_inc - rev_exc
                else:
                    # 电力销售/辅助服务增值税率 13%
                    rev_inc = discharge_revenue + ancillary_revenue
                    rev_exc = rev_inc / (1 + StorageConstants.VAT_ELECTRICITY)
                    output_vat = rev_inc - rev_exc

                df.loc[y, 'Charge_Cost'] = charge_cost
                df.loc[y, 'Discharge_Revenue'] = discharge_revenue
                df.loc[y, 'Lease_Revenue'] = lease_revenue
                df.loc[y, 'Ancillary_Revenue'] = ancillary_revenue
                df.loc[y, 'Revenue_Inc'] = rev_inc
                df.loc[y, 'Revenue_Exc'] = rev_exc
                df.loc[y, 'Output_VAT'] = output_vat

                # ========== 成本计算 ==========
                # 运维费
                om_cost = self._get_om_rate(op_year)
                df.loc[y, 'OM_Cost'] = om_cost

                # ========== 税务计算 ==========
                # 增值税抵扣池逻辑
                if current_deductible > 0:
                    if current_deductible >= output_vat:
                        current_deductible -= output_vat
                        vat_pay = 0
                    else:
                        vat_pay = output_vat - current_deductible
                        current_deductible = 0
                else:
                    vat_pay = output_vat

                df.loc[y, 'VAT_Payable'] = vat_pay
                surtax = vat_pay * StorageConstants.SURTAX_RATE
                df.loc[y, 'Surtax'] = surtax

                # ========== 电池更换费用 ==========
                # 在电池寿命到期年份产生更换费用
                battery_replacement = 0.0
                if op_year % self.battery_life == 0 and op_year < StorageConstants.OPERATION_PERIOD:
                    battery_replacement = self.replacement_cost
                df.loc[y, 'Battery_Replacement'] = battery_replacement

                # ========== 折旧 (分离电池与非电池资产) ==========
                # 电池折旧: 按电池寿命 (调峰10年, 调频4年)
                battery_depreciation = battery_depreciation_per_year if op_year <= self.battery_life else 0
                # 非电池折旧: 按15年折旧
                non_battery_depreciation = non_battery_depreciation_per_year if op_year <= StorageConstants.DEPRECIATION_YEARS_NON_BATTERY else 0
                depreciation = battery_depreciation + non_battery_depreciation
                df.loc[y, 'Depreciation'] = depreciation

                # ========== 利润与所得税 ==========
                # 利润总额 = 收入 - 充电成本 - 运维费 - 附加税 - 折旧 - 电池更换(费用化)
                if self.replacement_mode == StorageConstants.REPLACEMENT_EXPENSE:
                    profit = rev_exc - charge_cost - om_cost - surtax - depreciation - battery_replacement
                else:
                    # 资本化模式下，电池更换不作为当期费用
                    profit = rev_exc - charge_cost - om_cost - surtax - depreciation

                # 三免三减半政策 (储能项目可能享受)
                if op_year <= 3:
                    tax_rate = 0.0
                elif op_year <= 6:
                    tax_rate = StorageConstants.INCOME_TAX_RATE * 0.5
                else:
                    tax_rate = StorageConstants.INCOME_TAX_RATE

                income_tax = max(0.0, profit * tax_rate)
                df.loc[y, 'Income_Tax'] = income_tax

                # ========== 现金流合成 ==========
                # 现金流入 = 放电收入 + 租赁收入 + 辅助服务收入
                inflow = discharge_revenue + lease_revenue + ancillary_revenue

                # 最后一年回收余值和流动资金
                if y == StorageConstants.OPERATION_PERIOD + 1:
                    residual = self.static_invest * StorageConstants.RESIDUAL_RATIO
                    inflow += residual + working_capital

                # 现金流出 = 充电成本 + 运维费 + 附加税 + 电池更换 + 所得税
                outflow = charge_cost + om_cost + surtax + battery_replacement + income_tax

                df.loc[y, 'Net_CF_Pre'] = inflow - (charge_cost + om_cost + surtax + battery_replacement)
                df.loc[y, 'Net_CF_After'] = inflow - outflow

            self.df = df
            self.total_invest = total_invest
            self.const_interest = const_interest

            logger.info(f"现金流计算完成: 总投资={total_invest:.2f}万元")
            return df

        except Exception as e:
            raise CalculationError(f"现金流计算失败: {e}") from e

    def get_metrics(self) -> Dict[str, float]:
        """
        计算核心指标

        Returns:
            包含以下指标的字典:
                - 总投资 (万元)
                - 建设期利息 (万元)
                - 全投资IRR(税前) (%)
                - 全投资IRR(税后) (%)
                - 投资回收期 (年)
        """
        if self.df is None:
            raise CalculationError("请先运行 calculate_cash_flow()")

        try:
            cf_pre = self.df['Net_CF_Pre'].values
            cf_after = self.df['Net_CF_After'].values

            irr_pre = npf.irr(cf_pre) * 100
            irr_after = npf.irr(cf_after) * 100

            # 静态投资回收期计算
            cumsum = np.cumsum(cf_after)
            positive_indices = np.where(cumsum >= 0)[0]

            if len(positive_indices) > 0:
                p_idx = positive_indices[0]
                payback = p_idx - 1 + abs(cumsum[p_idx - 1]) / cf_after[p_idx] if p_idx > 0 else 1.0
            else:
                logger.warning("项目在运营期内无法收回投资")
                payback = 99.9

            return {
                "总投资": round(self.total_invest, 2),
                "建设期利息": round(self.const_interest, 2),
                "全投资IRR(税前)": round(irr_pre, 2),
                "全投资IRR(税后)": round(irr_after, 2),
                "投资回收期(年)": round(payback, 2)
            }

        except Exception as e:
            raise CalculationError(f"指标计算失败: {e}") from e

    # ==============================================================================
    # 财务报表输出方法
    # ==============================================================================

    def export_revenue_tax_table(self, filename: Optional[str] = None) -> pd.DataFrame:
        """
        导出收入和税金表

        Args:
            filename: 输出文件名
        """
        if self.df is None:
            raise CalculationError("请先运行 calculate_cash_flow()")

        cashflow_df = self.df[self.df.index >= 2].copy()

        table = pd.DataFrame({
            '年份': [f'第{i}年' for i in range(1, StorageConstants.OPERATION_PERIOD + 1)],
            '充电量(MWh)': [self.capacity_mwh] * StorageConstants.OPERATION_PERIOD,
            '放电量(MWh)': [self.capacity_mwh * self.efficiency] * StorageConstants.OPERATION_PERIOD,
            '营业收入(含税,万元)': cashflow_df['Revenue_Inc'].values,
            '营业收入(不含税,万元)': cashflow_df['Revenue_Exc'].values,
            '增值税(万元)': cashflow_df['Output_VAT'].values,
            '增值税实缴(万元)': cashflow_df['VAT_Payable'].values,
            '附加税(万元)': cashflow_df['Surtax'].values,
        })

        if filename:
            table.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"收入和税金表已保存到: {filename}")

        return table

    def export_total_cost_table(self, filename: Optional[str] = None) -> pd.DataFrame:
        """
        导出总成本费用估算表

        Args:
            filename: 输出文件名
        """
        if self.df is None:
            raise CalculationError("请先运行 calculate_cash_flow()")

        cashflow_df = self.df[self.df.index >= 2].copy()

        deductible_tax = self.p.get('deductible_tax', self.static_invest / (1 + StorageConstants.VAT_ELECTRICITY) * StorageConstants.VAT_ELECTRICITY)
        const_interest = self.const_interest
        fixed_asset_value = self.static_invest + const_interest - deductible_tax

        # 分离电池与非电池资产折旧计算
        battery_asset_value = fixed_asset_value * self.battery_asset_ratio
        non_battery_asset_value = fixed_asset_value * (1 - self.battery_asset_ratio)
        battery_depreciation_per_year = battery_asset_value * StorageConstants.DEPRECIATION_BASE_RATIO / self.battery_life
        non_battery_depreciation_per_year = non_battery_asset_value * StorageConstants.DEPRECIATION_BASE_RATIO / StorageConstants.DEPRECIATION_YEARS_NON_BATTERY

        # 计算每年的折旧额
        depreciation_list = []
        for i in range(1, StorageConstants.OPERATION_PERIOD + 1):
            battery_dep = battery_depreciation_per_year if i <= self.battery_life else 0
            non_battery_dep = non_battery_depreciation_per_year if i <= StorageConstants.DEPRECIATION_YEARS_NON_BATTERY else 0
            depreciation_list.append(battery_dep + non_battery_dep)

        table = pd.DataFrame({
            '年份': [f'第{i}年' for i in range(1, StorageConstants.OPERATION_PERIOD + 1)],
            '运维成本(万元)': cashflow_df['OM_Cost'].values,
            '电池更换费用(万元)': cashflow_df['Battery_Replacement'].values,
            '折旧费(万元)': depreciation_list,
            '摊销费(万元)': [0.0] * StorageConstants.OPERATION_PERIOD,
            '财务费用(万元)': [0.0] * StorageConstants.OPERATION_PERIOD,
            '总成本费用(万元)': cashflow_df['OM_Cost'].values + cashflow_df['Battery_Replacement'].values + depreciation_list,
        })

        table['经营成本(万元)'] = table['运维成本(万元)'] + table['电池更换费用(万元)']

        if filename:
            table.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"总成本费用表已保存到: {filename}")

        return table

    def export_profit_table(self, filename: Optional[str] = None) -> pd.DataFrame:
        """
        导出利润与利润分配表

        Args:
            filename: 输出文件名
        """
        if self.df is None:
            raise CalculationError("请先运行 calculate_cash_flow()")

        cashflow_df = self.df[self.df.index >= 2].copy()

        deductible_tax = self.p.get('deductible_tax', self.static_invest / (1 + StorageConstants.VAT_ELECTRICITY) * StorageConstants.VAT_ELECTRICITY)
        const_interest = self.const_interest
        fixed_asset_value = self.static_invest + const_interest - deductible_tax

        # 分离电池与非电池资产折旧计算
        battery_asset_value = fixed_asset_value * self.battery_asset_ratio
        non_battery_asset_value = fixed_asset_value * (1 - self.battery_asset_ratio)
        battery_depreciation_per_year = battery_asset_value * StorageConstants.DEPRECIATION_BASE_RATIO / self.battery_life
        non_battery_depreciation_per_year = non_battery_asset_value * StorageConstants.DEPRECIATION_BASE_RATIO / StorageConstants.DEPRECIATION_YEARS_NON_BATTERY

        profit_list = []
        depreciation_list = []
        for i in range(1, StorageConstants.OPERATION_PERIOD + 1):
            battery_dep = battery_depreciation_per_year if i <= self.battery_life else 0
            non_battery_dep = non_battery_depreciation_per_year if i <= StorageConstants.DEPRECIATION_YEARS_NON_BATTERY else 0
            depreciation = battery_dep + non_battery_dep
            depreciation_list.append(depreciation)
            profit = cashflow_df.loc[i + 1, 'Revenue_Exc'] - cashflow_df.loc[i + 1, 'Charge_Cost'] - cashflow_df.loc[i + 1, 'OM_Cost'] - cashflow_df.loc[i + 1, 'Surtax'] - depreciation - cashflow_df.loc[i + 1, 'Battery_Replacement']
            profit_list.append(profit)

        table = pd.DataFrame({
            '年份': [f'第{i}年' for i in range(1, StorageConstants.OPERATION_PERIOD + 1)],
            '营业收入(不含税,万元)': cashflow_df['Revenue_Exc'].values,
            '充电成本(万元)': cashflow_df['Charge_Cost'].values,
            '营业税金及附加(万元)': cashflow_df['Surtax'].values,
            '总成本费用(万元)': cashflow_df['OM_Cost'].values + cashflow_df['Battery_Replacement'].values + depreciation_list,
            '利润总额(万元)': profit_list,
            '所得税(万元)': cashflow_df['Income_Tax'].values,
            '净利润(万元)': [p - t for p, t in zip(profit_list, cashflow_df['Income_Tax'].values)],
        })

        table['累计净利润(万元)'] = table['净利润(万元)'].cumsum()

        if filename:
            table.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"利润表已保存到: {filename}")

        return table

    def export_financial_summary_table(self, filename: Optional[str] = None) -> pd.DataFrame:
        """
        导出财务指标汇总表

        Args:
            filename: 输出文件名
        """
        if self.df is None:
            raise CalculationError("请先运行 calculate_cash_flow()")

        metrics = self.get_metrics()
        cashflow_df = self.df[self.df.index >= 2].copy()

        total_profit = cashflow_df['Revenue_Exc'].sum() - cashflow_df['Charge_Cost'].sum() - cashflow_df['OM_Cost'].sum() - cashflow_df['Surtax'].sum()
        roi = total_profit / self.total_invest * 100

        table = pd.DataFrame({
            '指标': [
                '项目总投资(万元)',
                '建设期利息(万元)',
                '全投资IRR(税前,%)',
                '全投资IRR(税后,%)',
                '投资回收期(年)',
                '总投资收益率(ROI,%)',
                '年均净利润(万元)',
                f'{StorageConstants.OPERATION_PERIOD}年累计净利润(万元)',
                '装机功率(MW)',
                '额定容量(MWh)',
                '单位造价(元/Wh)',
                '系统能效(%)',
            ],
            '数值': [
                metrics['总投资'],
                metrics['建设期利息'],
                metrics['全投资IRR(税前)'],
                metrics['全投资IRR(税后)'],
                metrics['投资回收期(年)'],
                round(roi, 2),
                round(total_profit / StorageConstants.OPERATION_PERIOD, 2),
                round(total_profit, 2),
                self.power_mw,
                self.capacity_mwh,
                round(self.static_invest / (self.capacity_mwh * 1000), 2),
                round(self.efficiency * 100, 1),
            ],
        })

        if filename:
            table.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"财务指标汇总表已保存到: {filename}")

        return table


# ==============================================================================
# 敏感性分析
# ==============================================================================

def storage_sensitivity_analysis(
    base_params: Dict[str, Any],
    factor: str,
    variation_range: float = 0.10,
    steps: int = 5
) -> pd.DataFrame:
    """
    储能项目单因素敏感性分析

    Args:
        base_params: 基础项目参数
        factor: 要分析的因素
        variation_range: 变化范围
        steps: 分析步数

    Returns:
        敏感性分析结果 DataFrame
    """
    results = []
    base_value = base_params.get(factor)

    if base_value is None:
        raise ValueError(f"未知的因素: {factor}")

    variations = np.linspace(-variation_range, variation_range, steps)

    for var in variations:
        params_temp = base_params.copy()
        new_value = base_value * (1 + var)
        params_temp[factor] = new_value

        try:
            project = StorageProject(params_temp)
            project.calculate_cash_flow()
            metrics = project.get_metrics()
            irr = metrics['全投资IRR(税前)']

            results.append({
                '因素': factor,
                '变化率': f'{var*100:+.1f}%',
                '数值': new_value,
                'IRR(税前)%': irr,
            })
        except Exception as e:
            logger.error(f"敏感性分析失败 (变化率={var*100:.1f}%): {e}")
            results.append({
                '因素': factor,
                '变化率': f'{var*100:+.1f}%',
                '数值': new_value,
                'IRR(税前)%': None,
            })

    df = pd.DataFrame(results)
    logger.info(f"敏感性分析完成: 因素={factor}")
    return df


# ==============================================================================
# 演示与测试
# ==============================================================================

def demo_storage_project() -> None:
    """
    典型独立储能电站演示案例

    100MW/200MWh (2小时储能) 独立储能电站
    """
    print("\n" + "=" * 70)
    print("⚡ PyStorage-Eval - 电化学储能电站技经评价引擎")
    print("=" * 70)

    demo_params = {
        # 基础参数
        'power_mw': 100.0,                # 装机功率 100MW
        'capacity_mwh': 200.0,            # 额定容量 200MWh (2小时储能)
        'efficiency': 0.85,               # 系统效率 85%
        'static_invest': 30000.0,         # 静态投资 3亿元 (1.5元/Wh)
        'loan_rate': 0.048,
        'capital_ratio': 0.2,

        # 收益模式: 峰谷套利
        'revenue_mode': 'arbitrage',
        'cycles_per_year': 330,           # 年循环次数
        'charge_price': 0.3,             # 充电电价 0.3元/kWh
        'discharge_price': 0.9,          # 放电电价 0.9元/kWh

        # 电池更换策略
        'battery_life': 10,               # 调峰项目电池寿命 10年
        'replacement_cost': 21000.0,      # 电池更换成本 (静态投资的70%)
        'replacement_mode': 'expense',    # 费用化
    }

    try:
        print("\n📊 项目参数:")
        print(f"   装机功率: {demo_params['power_mw']} MW")
        print(f"   额定容量: {demo_params['capacity_mwh']} MWh")
        print(f"   系统效率: {demo_params['efficiency']:.0%}")
        print(f"   静态投资: {demo_params['static_invest']:.0f} 万元")
        print(f"   单位造价: {demo_params['static_invest'] / (demo_params['capacity_mwh'] * 1000):.2f} 元/Wh")
        print(f"   收益模式: 峰谷套利")
        print(f"   价差: {demo_params['discharge_price'] - demo_params['charge_price']} 元/kWh")
        print(f"   电池寿命: {demo_params['battery_life']} 年")

        project = StorageProject(demo_params)
        project.calculate_cash_flow()
        metrics = project.get_metrics()

        print("\n" + "-" * 70)
        print("✅ 独立储能电站技经评价报告")
        print("-" * 70)
        print(f"💰 项目总投资:      {metrics['总投资']:>15,.2f} 万元")
        print(f"🏗️  建设期利息:     {metrics['建设期利息']:>15,.2f} 万元")
        print(f"📈 IRR (税前):      {metrics['全投资IRR(税前)']:>15,.2f} %")
        print(f"📉 IRR (税后):      {metrics['全投资IRR(税后)']:>15,.2f} %")
        print(f"📅 投资回收期:      {metrics['投资回收期(年)']:>15,.2f} 年")
        print("-" * 70)

        # 现金流摘要
        df = project.df
        print(f"\n📈 {StorageConstants.OPERATION_PERIOD}年累计:")
        print(f"   放电收入: {df['Discharge_Revenue'].sum():,.2f} 万元")
        print(f"   充电成本: {df['Charge_Cost'].sum():,.2f} 万元")
        print(f"   运维费用: {df['OM_Cost'].sum():,.2f} 万元")
        print(f"   电池更换: {df['Battery_Replacement'].sum():,.2f} 万元")
        print(f"   净利润: {(df['Revenue_Exc'].sum() - df['Charge_Cost'].sum() - df['OM_Cost'].sum() - df['Surtax'].sum() - df['Depreciation'].sum() - df['Battery_Replacement'].sum()):,.2f} 万元")

        # 导出财务报表
        print("\n" + "=" * 70)
        print("📄 正在生成财务报表...")
        print("=" * 70)

        project.export_revenue_tax_table('output_收入和税金表.csv')
        print("✅ 收入和税金表: output_收入和税金表.csv")

        project.export_total_cost_table('output_总成本费用表.csv')
        print("✅ 总成本费用表: output_总成本费用表.csv")

        project.export_profit_table('output_利润表.csv')
        print("✅ 利润与利润分配表: output_利润表.csv")

        project.export_financial_summary_table('output_财务指标汇总表.csv')
        print("✅ 财务指标汇总表: output_财务指标汇总表.csv")

        # 敏感性分析
        print("\n" + "=" * 70)
        print("📈 正在进行敏感性分析...")
        print("=" * 70)

        for factor in ['static_invest', 'discharge_price', 'charge_price', 'cycles_per_year']:
            factor_names = {
                'static_invest': '静态投资',
                'discharge_price': '放电电价',
                'charge_price': '充电电价',
                'cycles_per_year': '循环次数'
            }
            sens_df = storage_sensitivity_analysis(demo_params, factor, variation_range=0.15, steps=5)
            filename = f'output_敏感性分析_{factor_names[factor]}.csv'
            sens_df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"✅ {factor_names[factor]}敏感性分析: {filename}")

        print("=" * 70)

    except (InputValidationError, CalculationError) as e:
        print(f"\n❌ 错误: {e}")
    except Exception as e:
        print(f"\n❌ 未知错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_storage_project()
