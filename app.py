# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from database.models import db, User, Trade, Forecast
from data.loader import get_hs300_data
from utils.indicators import add_indicators, simulate_order  # 添加simulate_order导入
from datetime import datetime
import pandas as pd
from predict import predict_future, create_features, prepare_sequences_multivariate
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


@app.route('/get_analysis', methods=['GET', 'POST'])
@login_required
def get_analysis():
    try:
        end_date = None
        if request.method == 'POST':
            json_data = request.json or {}
            end_date = json_data.get('end')
        df = get_hs300_data()
        df = create_features(df)
        if end_date:
            min_date = pd.to_datetime('2015-01-01')
            user_date = pd.to_datetime(end_date)
            df = df[(df['date'] >= min_date) & (df['date'] <= user_date)].reset_index(drop=True)
        future = predict_future(target_date=end_date)
        if not future:
            return jsonify(success=False, message="数据不足或预测失败")
        analysis_obj = generate_strategy(df, future)
        return jsonify(success=True, analysis=analysis_obj, future=future)
    except Exception as e:
        return jsonify(success=False, message=f"分析失败: {str(e)}")
@app.route('/valid_dates')
@login_required
def valid_dates():
    df = get_hs300_data()
    dates = sorted(df['date'].astype(str).tolist())
    return jsonify(dates=dates)
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
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    return render_template('simulate_trade.html', data=df.to_dict(orient='records'))


@app.route('/simulate_order', methods=['POST'])
@login_required
def simulate_order_route():
    try:
        data = request.json
        side = data.get('side')
        order_type = data.get('type')
        order_date = data.get('date')
        price = float(data.get('price')) if data.get('price') else None
        quantity = data.get('quantity')
        stop_loss = float(data.get('stopLoss')) if data.get('stopLoss') else None
        take_profit = float(data.get('takeProfit')) if data.get('takeProfit') else None
        close_date = data.get('closeDate')

        df = get_hs300_data()
        # 买入
        if side == 'buy':
            if not quantity or int(quantity) <= 0:
                return jsonify(success=False, message="请填写有效股数")
            quantity = int(quantity)
            buy_price = price or df[df['date'] == order_date]['close'].iloc[0]
            total_cost = buy_price * quantity
            if current_user.cash < Decimal(str(total_cost)):
                return jsonify(success=False, message="账户现金不足，无法完成买入")
            # 允许多持仓，不限持仓数
            trade = Trade(user_id=current_user.id,
                buy_date=pd.to_datetime(order_date),
                buy_price=Decimal(str(buy_price)),
                quantity=quantity,
                sell_date=None,
                sell_price=None,
                profit=None,
                profit_rate=None,
                status='filled',
                order_type=order_type,
                side=side
            )
            db.session.add(trade)
            current_user.holding_quantity += quantity
            current_user.cash -= Decimal(str(total_cost))
            db.session.commit()
            info = {
                'buy_date': order_date,
                'quantity': quantity,
                'buy_price': buy_price,
                'sell_date': '持仓中',
                'sell_price': '--',
                'profit': '--',
                'profit_rate': '--',
                'is_simulation': False
            }
            return jsonify(success=True, chart_data={}, buy_point=[0, float(buy_price)], sell_point=None, info=info)
        # 卖出
        elif side == 'sell':
            # 取最早未卖出持仓
            trades = Trade.query.filter_by(user_id=current_user.id, sell_price=None, status='filled').order_by(Trade.buy_date).all()
            if not trades or len(trades) == 0:
                return jsonify(success=False, message="没有可卖出的持仓记录")
            latest_trade = trades[0]

            if not close_date:
                return jsonify(success=False, message="请设置卖出日期")
            sell_price = price or df[df['date'] == close_date]['close'].iloc[0]
            profit = (sell_price - float(latest_trade.buy_price)) * latest_trade.quantity
            denominator = float(latest_trade.buy_price) * latest_trade.quantity
            profit_rate = (profit / denominator) * 100 if denominator != 0 else 0

            latest_trade.sell_date = pd.to_datetime(close_date)
            latest_trade.sell_price = Decimal(str(sell_price))
            latest_trade.profit = Decimal(str(profit))
            latest_trade.profit_rate = Decimal(str(profit_rate))
            current_user.holding_quantity -= latest_trade.quantity if current_user.holding_quantity >= latest_trade.quantity else 0
            current_user.cash += Decimal(str(sell_price * latest_trade.quantity))
            db.session.commit()
            info = {
                'buy_date': latest_trade.buy_date.strftime('%Y-%m-%d'),
                'quantity': latest_trade.quantity,
                'buy_price': float(latest_trade.buy_price),
                'sell_date': close_date,
                'sell_price': float(latest_trade.sell_price),
                'profit': float(latest_trade.profit),
                'profit_rate': float(latest_trade.profit_rate),
                'is_simulation': False
            }
            chart_data = {
                'dates': [latest_trade.buy_date.strftime('%Y-%m-%d'), close_date],
                'prices': [float(latest_trade.buy_price), float(sell_price)]
            }
            return jsonify(success=True, chart_data=chart_data, buy_point=[0, float(latest_trade.buy_price)], sell_point=[1, float(sell_price)], info=info)
        else:
            return jsonify(success=False, message="无效的交易方向")
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