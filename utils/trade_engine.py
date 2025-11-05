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

        if direction == 'BUY':
            if order_type == 'MARKET':
                total_cost = current_price * quantity
            else:
                total_cost = price * quantity

            if user.cash < total_cost:
                return False, "资金不足"

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

            db.session.add(trade)

            if order_type == 'MARKET':
                user.cash -= total_cost

            db.session.commit()
            return True, "订单提交成功"

        elif direction == 'SELL':
            # 检查持仓
            position = TradeEngine.get_position(user_id)
            if position < quantity:
                return False, f"持仓不足，当前持仓: {position}股"

            buy_trades = Trade.query.filter_by(
                user_id=user_id,
                sell_price=None,
                status='filled'
            ).order_by(Trade.buy_date).all()

            if order_type == 'MARKET':
                remaining = quantity
                for buy_trade in buy_trades:
                    if remaining <= 0:
                        break
                    sell_qty = min(remaining, buy_trade.quantity)
                    if sell_qty < buy_trade.quantity:
                        # 分割剩余持仓
                        remaining_buy = Trade(
                            user_id=user_id,
                            buy_date=buy_trade.buy_date,
                            buy_price=buy_trade.buy_price,
                            quantity=buy_trade.quantity - sell_qty,
                            order_type=buy_trade.order_type,
                            limit_price=buy_trade.limit_price,
                            stop_loss=buy_trade.stop_loss,
                            take_profit=buy_trade.take_profit,
                            status='filled'
                        )
                        db.session.add(remaining_buy)
                        buy_trade.quantity = sell_qty

                    buy_trade.sell_date = datetime.now().date()
                    buy_trade.sell_price = current_price
                    remaining -= sell_qty

                if remaining > 0:
                    return False, "持仓不足"

                user.cash += current_price * quantity
                db.session.commit()
                return True, "订单提交成功"
            else:
                # LIMIT sell, 简化不实现完整限价逻辑
                return False, "限价卖出暂不支持"

    @staticmethod
    def get_position(user_id):
        """获取持仓数量 - 只计算未卖出的买入交易"""
        buys = db.session.query(db.func.sum(Trade.quantity)).filter(
            Trade.user_id == user_id,
            Trade.sell_price == None,
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