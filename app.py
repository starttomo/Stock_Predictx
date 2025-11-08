# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from database.models import db, User, Trade, Forecast
from data.loader import get_hs300_data
from utils.indicators import add_indicators, simulate_order  # 添加simulate_order导入
from datetime import datetime
import pandas as pd
from  analysis.strategy import generate_strategy
from decimal import Decimal
from utils.trade_engine import TradeEngine
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:zhang728@localhost/stock_system'
db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# app.py - 添加以下路由
from utils.trade_engine import TradeEngine


@app.route('/get_analysis')
@login_required
def get_analysis():
    df = get_hs300_data()
    df = add_indicators(df)

    # 获取预测用于策略
    forecasts = Forecast.query.order_by(Forecast.date).all()
    future = [(f.date.strftime('%Y-%m-%d'), float(f.yhat)) for f in forecasts[-5:]]  # future_predictions格式：list of (date, yhat)

    analysis = generate_strategy(df, future)  # 传入future
    return jsonify(analysis=analysis)


@app.route('/get_forecast')
@login_required
def get_forecast():
    forecasts = Forecast.query.order_by(Forecast.date).all()
    future_data = [{
        'date': f.date.strftime('%Y-%m-%d'),
        'yhat': float(f.yhat),
        'yhat_lower': float(f.yhat_lower),
        'yhat_upper': float(f.yhat_upper)
    } for f in forecasts[-5:]]
    return jsonify(future_data=future_data)


@app.route('/place_order', methods=['POST'])
@login_required
def place_order():
    data = request.json
    direction = data.get('direction', 'BUY')
    order_type = data.get('order_type', 'MARKET')
    quantity = int(data.get('quantity', 0))
    price = Decimal(str(data.get('price', 0))) if data.get('price') else None

    if quantity % 100 != 0:
        return jsonify(success=False, message="必须100股的倍数")

    success, message = TradeEngine.place_order(
        user_id=current_user.id,
        direction=direction,
        order_type=order_type,
        quantity=quantity,
        price=price
    )

    if success:
        portfolio = TradeEngine.get_portfolio_value(current_user.id)
        return jsonify(success=True, message=message, portfolio=portfolio)
    else:
        return jsonify(success=False, message=message)


@app.route('/portfolio')
@login_required
def get_portfolio():
    portfolio = TradeEngine.get_portfolio_value(current_user.id)
    return jsonify(portfolio)


@app.route('/order_history')
@login_required
def order_history():
    trades = Trade.query.filter_by(user_id=current_user.id).order_by(Trade.created_at.desc()).all()
    return render_template('order_history.html', trades=trades)
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and user.password == request.form['password']:
            login_user(user)
            return redirect(url_for('index'))
        flash('登录失败')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        if User.query.filter_by(username=request.form['username']).first():
            flash('用户已存在')
        else:
            user = User(username=request.form['username'], password=request.form['password'])
            db.session.add(user)
            db.session.commit()
            flash('注册成功')
            return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/')
@login_required
def index():
    df = get_hs300_data()
    df = add_indicators(df)
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    # 预测
    forecasts = Forecast.query.order_by(Forecast.date).all()
    future = [(f.date.strftime('%Y-%m-%d'), float(f.yhat)) for f in forecasts[-5:]]

    # 持仓计算（不变）
    trades = Trade.query.filter_by(user_id=current_user.id, sell_price=None).all()
    total_shares = sum(t.quantity for t in trades)
    total_cost = sum(t.quantity * float(t.buy_price) for t in trades)
    avg_price = total_cost / total_shares if total_shares > 0 else 0
    current_price = float(df['close'].iloc[-1])
    market_value = total_shares * current_price

    # 资产更新（不变）
    from decimal import Decimal
    total_assets = current_user.cash + Decimal(str(market_value))
    profit = total_assets - current_user.total_invest
    profit_rate = float(profit / current_user.total_invest * 100) if current_user.total_invest > 0 else 0

    current_user.current_value = total_assets
    current_user.profit = profit
    current_user.profit_rate = profit_rate
    db.session.commit()

    # 生成策略分析 - 传入future
    analysis = generate_strategy(df, future)

    return render_template('index.html',
                           data=df.to_dict(orient='records'),
                           future=future,
                           position={'shares': total_shares, 'avg_price': avg_price},
                           current_price=current_price,
                           total_assets=total_assets,
                           cash=current_user.cash,
                           profit=profit,
                           profit_rate=profit_rate,
                           analysis=analysis,
                           today=datetime.now().strftime('%Y-%m-%d'))
@app.route('/simulate_trade')
@login_required
def simulate_trade():
    df = get_hs300_data()
    df = add_indicators(df)
    return render_template('simulate_trade.html', data=df.to_dict(orient='records'))

# 在app.py中添加缺失的路由
@app.route('/predict_guide', methods=['POST'])
@login_required
def predict_guide():
    """预测指导接口"""
    try:
        data = request.json
        start_date = data.get('start')
        end_date = data.get('end')

        if not start_date or not end_date:
            return jsonify(success=False, message="请选择起始和结束日期")

        df = get_hs300_data()
        df = add_indicators(df)

        # 过滤指定日期范围的数据
        mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
        period_df = df[mask]

        if period_df.empty:
            return jsonify(success=False, message="所选日期范围内无数据")

        # 生成简单的预测指导
        latest = period_df.iloc[-1]
        guide = f"""
        <h4>预测指导分析 ({start_date} 至 {end_date})</h4>
        <p><strong>收盘价:</strong> {latest['close']:.2f}</p>
        <p><strong>RSI:</strong> {latest['rsi']:.2f} - {'超买' if latest['rsi'] > 70 else '超卖' if latest['rsi'] < 30 else '中性'}</p>
        <p><strong>价格位置:</strong> {'上轨附近' if latest['close'] >= latest['upper'] else '下轨附近' if latest['close'] <= latest['lower'] else '轨道中部'}</p>
        <p><strong>均线趋势:</strong> {'多头排列' if latest['ma7'] > latest['ma30'] else '空头排列'}</p>
        """

        return jsonify(success=True, guide=guide)

    except Exception as e:
        return jsonify(success=False, message=f"预测指导失败: {str(e)}")

# 修复simulate_order路由中的函数调用问题
@app.route('/simulate_order', methods=['POST'])
@login_required
def simulate_order_route():  # 重命名函数避免冲突
    try:
        data = request.json
        side = data.get('side')
        order_type = data.get('type')
        order_date = data.get('date')
        price = float(data.get('price')) if data.get('price') else None
        quantity = int(data.get('quantity'))
        stop_loss = float(data.get('stopLoss')) if data.get('stopLoss') else None
        take_profit = float(data.get('takeProfit')) if data.get('takeProfit') else None
        close_date = data.get('closeDate')

        if not order_date:
            return jsonify(success=False, message="请选择挂单日期")

        df = get_hs300_data()

        # 检查买入卖出逻辑
        if side == 'buy':
            # 买入逻辑
            if current_user.holding_quantity > 0:
                return jsonify(success=False, message="您已有持仓，请先卖出再买入")

            # 如果未选择平仓日期，则显示从买入到当前时间的走势
            if not close_date:
                close_date = datetime.now().strftime('%Y-%m-%d')
                is_simulation_only = True  # 仅模拟，不保存到数据库
            else:
                is_simulation_only = False

        elif side == 'sell':
            # 卖出逻辑
            if current_user.holding_quantity <= 0:
                return jsonify(success=False, message="您没有持仓，无法卖出")

            if not close_date:
                return jsonify(success=False, message="卖出必须选择平仓日期")

            # 检查卖出日期是否在买入日期之后
            latest_trade = Trade.query.filter_by(user_id=current_user.id, sell_price=None).order_by(Trade.buy_date.desc()).first()
            if latest_trade and pd.to_datetime(close_date) <= latest_trade.buy_date:
                return jsonify(success=False, message="卖出日期必须在买入日期之后")

            is_simulation_only = False
        else:
            return jsonify(success=False, message="无效的交易方向")

        # 调用正确的simulate_order函数
        success, result = simulate_order(df, order_date, close_date, price, stop_loss, take_profit)

        if not success:
            return jsonify(success=False, message=result)

        # 解析返回结果
        chart_data, sell_date, sell_price = result

        # 计算盈亏
        buy_price = price or df[df['date'] == pd.to_datetime(order_date)]['close'].iloc[0]
        profit = (sell_price - buy_price) * quantity
        profit_rate = (profit / (buy_price * quantity)) * 100

        # 如果是买入且仅模拟（未选择平仓日期），则不保存到数据库
        if side == 'buy' and is_simulation_only:
            info = {
                'buy_date': order_date,
                'quantity': quantity,
                'buy_price': buy_price,
                'sell_date': '持仓中',
                'sell_price': '--',
                'profit': '--',
                'profit_rate': '--',
                'is_simulation': True
            }

            return jsonify(success=True,
                          chart_data=chart_data,
                          buy_point=[0, float(buy_price)],
                          sell_point=None,
                          info=info)

        # 将交易记录保存到数据库
        if side == 'buy':
            # 买入操作
            trade = Trade(
                user_id=current_user.id,
                buy_date=pd.to_datetime(order_date),
                buy_price=Decimal(str(buy_price)),
                quantity=quantity,
                sell_date=None,  # 买入时卖出日期为空
                sell_price=None,
                profit=None,
                profit_rate=None,
                status='filled',
                order_type=order_type,
                side=side
            )
            db.session.add(trade)

            # 更新用户持仓和现金
            current_user.holding_quantity += quantity
            current_user.cash -= Decimal(str(buy_price * quantity))

        elif side == 'sell':
            # 卖出操作 - 更新现有的持仓记录
            latest_trade = Trade.query.filter_by(user_id=current_user.id, sell_price=None).order_by(Trade.buy_date.desc()).first()
            if latest_trade:
                latest_trade.sell_date = pd.to_datetime(sell_date)
                latest_trade.sell_price = Decimal(str(sell_price))
                latest_trade.profit = Decimal(str(profit))
                latest_trade.profit_rate = Decimal(str(profit_rate))

                # 更新用户持仓和现金
                current_user.holding_quantity -= quantity
                current_user.cash += Decimal(str(sell_price * quantity))

        # 更新用户总资产和收益
        current_user.current_value = current_user.cash + Decimal(str(current_user.holding_quantity * float(df['close'].iloc[-1])))
        current_user.profit = current_user.current_value - current_user.total_invest
        current_user.profit_rate = float((current_user.profit / current_user.total_invest) * 100) if current_user.total_invest > 0 else 0

        db.session.commit()

        info = {
            'buy_date': order_date,
            'quantity': quantity,
            'buy_price': buy_price,
            'sell_date': sell_date.strftime('%Y-%m-%d'),
            'sell_price': sell_price,
            'profit': profit,
            'profit_rate': profit_rate,
            'is_simulation': False
        }

        return jsonify(success=True,
                      chart_data=chart_data,
                      buy_point=[0, float(buy_price)],
                      sell_point=[len(chart_data['dates']) - 1, float(sell_price)],
                      info=info)

    except Exception as e:
        return jsonify(success=False, message=f"模拟交易失败: {str(e)}")

@app.route('/profile')
@login_required
def profile():
    trades = Trade.query.filter_by(user_id=current_user.id).all()
    return render_template('profile.html', user=current_user, trades=trades)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    date = request.json['date']
    # 调用模型预测
    return jsonify({"price": 5200.0, "chart": [...]})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)