# PyPortfolioOpt 集成
将PyPortfolioOpt中的一些组合优化架构通过继承供YAML文件配置，

具体继承方式见 https://qlib.readthedocs.io/en/latest/component/strategy.html#base-class-interface 其中的
BaseStrategy类或WeightStrategyBase类

## Qlib base-class-interface文档部分
### BaseStrategy
Qlib provides a base class qlib.strategy.base.BaseStrategy. All strategy classes need to inherit the base class and implement its interface.
Qlib 提供了一个基础类 qlib.strategy.base.BaseStrategy 。所有策略类都需要继承这个基础类并实现其接口。

generate_trade_decision
generate_trade_decision is a key interface that generates trade decisions in each trading bar. The frequency to call this method depends on the executor frequency(“time_per_step”=”day” by default). But the trading frequency can be decided by users’ implementation. For example, if the user wants to trading in weekly while the time_per_step is “day” in executor, user can return non-empty TradeDecision weekly(otherwise return empty like this ).
generate_trade_decision 是一个关键接口，在每个交易栏中生成交易决策。调用此方法的频率取决于执行者频率（默认为“time_per_step”=“day”）。但交易频率可以由用户的实现决定。例如，如果用户希望在执行者中“time_per_step”为“day”的情况下按周进行交易，用户可以按周返回非空的 TradeDecision（否则像这样返回空）。

Users can inherit BaseStrategy to customize their strategy class.
用户可以继承 BaseStrategy 来自定义他们的策略类。

### WeightStrategyBase
Qlib also provides a class qlib.contrib.strategy.WeightStrategyBase that is a subclass of BaseStrategy.
Qlib 还提供了一个类 qlib.contrib.strategy.WeightStrategyBase ，它是 BaseStrategy 的子类。

WeightStrategyBase only focuses on the target positions, and automatically generates an order list based on positions. It provides the generate_target_weight_position interface.
WeightStrategyBase 只关注目标仓位，并自动根据仓位生成订单列表。它提供了 generate_target_weight_position 接口。

generate_target_weight_position
According to the current position and trading date to generate the target position. The cash is not considered in the output weight distribution.
根据当前位置和交易日期生成目标位置。输出权重分布时不考虑现金。

Return the target position.
返回目标仓位。

**Note  注意**

Here the target position means the target percentage of total assets.
这里的目标位置是指总资产的目标百分比。

WeightStrategyBase implements the interface generate_order_list, whose processions is as follows.
WeightStrategyBase 实现了 generate_order_list 接口，其处理过程如下。

Call generate_target_weight_position method to generate the target position.
调用 generate_target_weight_position 方法来生成目标仓位。

Generate the target amount of stocks from the target position.
从目标仓位生成目标股票数量。

Generate the order list from the target amount
从目标金额生成订单列表

Users can inherit WeightStrategyBase and implement the interface generate_target_weight_position to customize their strategy class, which only focuses on the target positions.
用户可以继承 WeightStrategyBase 并实现 generate_target_weight_position 接口来定制他们的策略类，该策略类只关注目标仓位。