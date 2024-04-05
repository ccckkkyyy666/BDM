import os
import pandas as pd
import argparse

from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from stable_baselines3 import PPO
from finrl.main import check_and_make_directories
from finrl.config import INDICATORS, TRAINED_MODEL_DIR
from finrl.plot import backtest_stats

# Multiple agents switching model designed by us
from ppo_switch import PPO_Switch


DIR = './data/ValidationSet/'

TRADE_START_DATE = '2009-07-02'
TRADE_END_DATE = '2010-07-02'
DATA_NAME = 'full-DOW29_validate'

# TRADE_START_DATE = '2018-02-05'
# TRADE_END_DATE = '2019-02-02'
# DATA_NAME = 'full-FTSE29_validate'

# TRADE_START_DATE = '2018-11-27'
# TRADE_END_DATE = '2019-11-27'
# DATA_NAME = 'full-HK29_validate'

# TRADE_START_DATE = '2012-02-06'
# TRADE_END_DATE = '2013-02-02'
# DATA_NAME = 'full-NYSE29_validate'

# TRADE_START_DATE = '2019-10-07'
# TRADE_END_DATE = '2020-10-07'
# DATA_NAME = 'full-CRYPTO29_validate'

# TRADE_START_DATE = '2010-01-05'
# TRADE_END_DATE = '2011-01-05'
# DATA_NAME = 'full-HS29_validate'

FILE_PATH = DIR + DATA_NAME + '.csv'


# PPO configs
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.0003,
    "batch_size": 128,
}

if __name__ == '__main__':
    # We will use unseen, post-deadline data for testing
    parser = argparse.ArgumentParser(description='Description of program')
    parser.add_argument('--start_date', default=TRADE_START_DATE,
                        help='Trade start date (default: {})'.format(TRADE_START_DATE))
    parser.add_argument('--end_date', default=TRADE_END_DATE,
                        help='Trade end date (default: {})'.format(TRADE_END_DATE))
    parser.add_argument('--data_file', default=FILE_PATH, help='Trade data file')

    args = parser.parse_args()
    TRADE_START_DATE = args.start_date
    TRADE_END_DATE = args.end_date

    processed_full = pd.read_csv(args.data_file)
    trade = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)

    stock_dimension = len(trade.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    buy_cost_list = sell_cost_list = [0.001] * stock_dimension  # fee
    num_stock_shares = [0] * stock_dimension

    # please do not change initial_amount
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }

    check_and_make_directories([TRAINED_MODEL_DIR])

    # Environment
    e_trade_gym_switch = StockTradingEnv(df=trade, **env_kwargs)
    e_trade_gym_real = StockTradingEnv(df=trade, **env_kwargs)
    e_trade_gym_max = StockTradingEnv(df=trade, **env_kwargs)
    e_trade_gym_min = StockTradingEnv(df=trade, **env_kwargs)
    e_trade_gym_mean = StockTradingEnv(df=trade, **env_kwargs)
    e_trade_gym_ema = StockTradingEnv(df=trade, **env_kwargs)

    # PPO agent
    # ppo_real model is trained with the real train datas
    ppo_real = PPO.load(TRAINED_MODEL_DIR + '/ppo_real')

    # ppo_max, ppo_min, ppo_mean and ppo_ema models are trained using four fake training datas respectively,
    # which is generated based on real data
    ppo_max = PPO.load(TRAINED_MODEL_DIR + '/ppo_max')
    ppo_min = PPO.load(TRAINED_MODEL_DIR + '/ppo_min')
    ppo_mean = PPO.load(TRAINED_MODEL_DIR + '/ppo_mean')
    ppo_ema = PPO.load(TRAINED_MODEL_DIR + '/ppo_ema')

    # Backtesting for d
    # for d in [1, 2, 3, 4, 5, 6, 7]:
    #     ppo_switch = PPO_Switch(alpha=0.5, switchWindows=[d], hmax=env_kwargs['hmax'],
    #                             stocksDimension=stock_dimension)
    #     df_result_ppo, df_actions_ppo = ppo_switch.DRL_prediction(
    #         model=[ppo_real, ppo_max, ppo_min, ppo_mean, ppo_ema],
    #         environment=[e_trade_gym_switch, e_trade_gym_real, e_trade_gym_max,
    #                      e_trade_gym_min, e_trade_gym_mean, e_trade_gym_ema])
    #
    #     print("==============Get Backtest Results===========")
    #     perf_stats_all = backtest_stats(account_value=df_result_ppo)
    #
    #     # save result
    #     save_dir = "./dResults/" + str(DATA_NAME[:-9]) + '/'
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #
    #     df_result_ppo.to_csv(save_dir + str(d) + "_d_cw.csv", index=False)

    # Backtesting for alpha
    for a in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        ppo_switch = PPO_Switch(alpha=a, switchWindows=[5], hmax=env_kwargs['hmax'],
                                stocksDimension=stock_dimension)
        df_result_ppo, df_actions_ppo = ppo_switch.DRL_prediction(
            model=[ppo_real, ppo_max, ppo_min, ppo_mean, ppo_ema],
            environment=[e_trade_gym_switch, e_trade_gym_real, e_trade_gym_max,
                         e_trade_gym_min, e_trade_gym_mean, e_trade_gym_ema])

        print("==============Get Backtest Results===========")
        perf_stats_all = backtest_stats(account_value=df_result_ppo)

        # save result
        save_dir = "./alphaResults/" + str(DATA_NAME[:-9]) + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        df_result_ppo.to_csv(save_dir + str(a) + "_alpha_cw.csv", index=False)
