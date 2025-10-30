# utils/trade_engine.py
from datetime import datetime
from decimal import Decimal
from database.models import db, Trade, User


class TradeEngine:
    @staticmethod
    def place_order(user_id, direction, order_type, quantity, price=None,
                    stop_loss=None, take_profit=None):
        """下订单"""
        user = User.query.get(user_id)

        # 获取当前价格数据
        from data.loader import get_hs300_data
        df = get_hs300_data()
        current_price = Decimal(str(df['close'].iloc[-1]))

        # 资金和持仓检查
        if direction == 'BUY':
            if order_type == 'MARKET':
                total_cost = current_price * quantity
            else:
                total_cost = price * quantity

            if user.cash < total_cost:
                return False, "资金不足"
        else:  # SELL 操作
            # 检查持仓是否足够
            position = TradeEngine.get_position(user_id)
            if position < quantity:
                return False, f"持仓不足，当前持仓: {position}股"

        # 创建订单
        if direction == 'BUY':
            trade = Trade(
                user_id=user_id,
                buy_date=datetime.now().date(),
                buy_price=price if price else current_price,
                quantity=quantity,
                order_type=order_type.lower(),
                limit_price=price if order_type == 'LIMIT' else None,
                stop_loss=stop_loss,
                take_profit=take_profit,
                status='filled' if order_type == 'MARKET' else 'pending'
            )

            # 市价单立即成交并扣款
            if order_type == 'MARKET':
                user.cash -= current_price * quantity

        else:  # SELL 操作
            # 查找要卖出的持仓
            buy_trades = Trade.query.filter_by(
                user_id=user_id,
                sell_price=None,
                status='filled'
            ).order_by(Trade.buy_date).all()

            # 计算可卖出数量
            available_shares = sum(t.quantity for t in buy_trades)
            if available_shares < quantity:
                return False, f"可卖出持仓不足，可用: {available_shares}股"

            # 创建卖出交易记录
            # 简化处理：平均卖出价格
            trade = Trade(
                user_id=user_id,
                buy_date=datetime.now().date(),  # 卖出时也记录为buy_date，但用direction区分
                buy_price=current_price,  # 卖出价格记录在buy_price字段
                quantity=quantity,
                order_type=order_type.lower(),
                limit_price=price if order_type == 'LIMIT' else None,
                stop_loss=stop_loss,
                take_profit=take_profit,
                status='filled' if order_type == 'MARKET' else 'pending'
            )

            # 标记为卖出交易（通过设置sell_price）
            trade.sell_price = current_price
            trade.sell_date = datetime.now().date()

            # 市价单立即成交并收款
            if order_type == 'MARKET':
                user.cash += current_price * quantity

        db.session.add(trade)
        db.session.commit()
        return True, "订单提交成功"

    @staticmethod
    def get_position(user_id):
        """获取持仓数量 - 只计算未卖出的买入交易"""
        buys = db.session.query(db.func.sum(Trade.quantity)).filter(
            Trade.user_id == user_id,
            Trade.sell_price == None,  # 未卖出的
            Trade.status == 'filled'
        ).scalar() or 0

        return buys

    @staticmethod
    def get_portfolio_value(user_id):
        """获取投资组合价值"""
        from data.loader import get_hs300_data
        df = get_hs300_data()
        current_price = Decimal(str(df['close'].iloc[-1]))

        user = User.query.get(user_id)
        position = TradeEngine.get_position(user_id)

        stock_value = position * current_price
        total_value = user.cash + stock_value

        return {
            'cash': float(user.cash),
            'position': position,
            'stock_value': float(stock_value),
            'total_value': float(total_value),
            'current_price': float(current_price)
        }

    @staticmethod
    def get_sellable_position(user_id):
        """获取可卖出持仓"""
        return TradeEngine.get_position(user_id)