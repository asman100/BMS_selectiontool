from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import pandas as pd
import pulp
import io
import os
import math

# --- Flask App and Login Configuration ---
app = Flask(__name__)
# This secret key is needed for session management
app.config['SECRET_KEY'] = 'a-very-secret-key-that-should-be-changed' 
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' # Redirect to 'login' view if user is not authenticated

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Default spare points percentage (can be overridden by user input)
DEFAULT_SPARE_POINTS_PERCENTAGE = 20

# --- User Management ---
# In a real-world application, this would be a database.
users = {'gila': {'password': 'BMS-gila22'}}

class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    if user_id in users:
        return User(user_id)
    return None

# --- Helper Functions (find_optimal_combination, get_accessories_recursively) ---
# These functions are unchanged from the previous version.
def find_optimal_combination(panel_requirements, components_df, spare_points_percentage=None):
    if spare_points_percentage is None:
        spare_points_percentage = DEFAULT_SPARE_POINTS_PERCENTAGE
    spare_multiplier = 1 + (spare_points_percentage / 100)
    
    prob = pulp.LpProblem(f"BMS_Combination_{panel_requirements['PanelName']}", pulp.LpMinimize)
    component_names = components_df['Name'].tolist(); component_qty_vars = pulp.LpVariable.dicts("Qty", component_names, lowBound=0, cat='Integer')
    components_map = components_df.set_index('Name').to_dict('index')
    # UIO points: first split into input vs output, then each splits into digital vs analog
    uio_as_di_vars = pulp.LpVariable.dicts("UIO_as_DI", component_names, lowBound=0, cat='Continuous')
    uio_as_ai_vars = pulp.LpVariable.dicts("UIO_as_AI", component_names, lowBound=0, cat='Continuous')
    uio_as_do_vars = pulp.LpVariable.dicts("UIO_as_DO", component_names, lowBound=0, cat='Continuous')
    uio_as_ao_vars = pulp.LpVariable.dicts("UIO_as_AO", component_names, lowBound=0, cat='Continuous')
    # UI points: split into digital vs analog
    ui_as_digital_vars = pulp.LpVariable.dicts("UI_as_Digital", component_names, lowBound=0, cat='Continuous')
    ui_as_analog_vars = pulp.LpVariable.dicts("UI_as_Analog", component_names, lowBound=0, cat='Continuous')
    # UO points: split into digital vs analog
    uo_as_digital_vars = pulp.LpVariable.dicts("UO_as_Digital", component_names, lowBound=0, cat='Continuous')
    uo_as_analog_vars = pulp.LpVariable.dicts("UO_as_Analog", component_names, lowBound=0, cat='Continuous')
    prob += pulp.lpSum([components_map[name]['Cost'] * component_qty_vars[name] for name in component_names]), "Total_Component_Cost"
    for name in component_names:
        safe_name = ''.join(e for e in name if e.isalnum())
        # UIO constraint: total UIO usage across all four types cannot exceed available
        available_uio = components_map[name].get('UIO', 0) * component_qty_vars[name]
        prob += uio_as_di_vars[name] + uio_as_ai_vars[name] + uio_as_do_vars[name] + uio_as_ao_vars[name] <= available_uio, f"UIO_Allocation_{safe_name}"
        # UI constraint: can be used for digital OR analog input (not both)
        available_ui = components_map[name].get('UI', 0) * component_qty_vars[name]
        prob += ui_as_digital_vars[name] + ui_as_analog_vars[name] <= available_ui, f"UI_Allocation_{safe_name}"
        # UO constraint: can be used for digital OR analog output (not both)
        available_uo = components_map[name].get('UO', 0) * component_qty_vars[name]
        prob += uo_as_digital_vars[name] + uo_as_analog_vars[name] <= available_uo, f"UO_Allocation_{safe_name}"
    
    # Apply spare points and round up (no decimals)
    required_di = math.ceil(panel_requirements.get('DI', 0) * spare_multiplier)
    required_do = math.ceil(panel_requirements.get('DO', 0) * spare_multiplier)
    required_ai = math.ceil(panel_requirements.get('AI', 0) * spare_multiplier)
    required_ao = math.ceil(panel_requirements.get('AO', 0) * spare_multiplier)
    
    # Individual constraints for each point type (Digital and Analog separately)
    provided_di = pulp.lpSum([components_map[name].get('DI', 0) * component_qty_vars[name] for name in component_names])
    provided_do = pulp.lpSum([components_map[name].get('DO', 0) * component_qty_vars[name] for name in component_names])
    provided_ai = pulp.lpSum([components_map[name].get('AI', 0) * component_qty_vars[name] for name in component_names])
    provided_ao = pulp.lpSum([components_map[name].get('AO', 0) * component_qty_vars[name] for name in component_names])
    
    # Universal inputs/outputs can be used for either digital or analog (but not both simultaneously)
    # UI and UIO can both contribute to digital or analog inputs
    provided_ui_digital = pulp.lpSum(ui_as_digital_vars) + pulp.lpSum(uio_as_di_vars)
    provided_ui_analog = pulp.lpSum(ui_as_analog_vars) + pulp.lpSum(uio_as_ai_vars)
    # UO and UIO can both contribute to digital or analog outputs
    provided_uo_digital = pulp.lpSum(uo_as_digital_vars) + pulp.lpSum(uio_as_do_vars)
    provided_uo_analog = pulp.lpSum(uo_as_analog_vars) + pulp.lpSum(uio_as_ao_vars)
    
    # Constraints: Each point type must be satisfied (with universal points allocated to specific types)
    prob += provided_di + provided_ui_digital >= required_di, "Digital_Input_Requirement"
    prob += provided_do + provided_uo_digital >= required_do, "Digital_Output_Requirement"
    prob += provided_ai + provided_ui_analog >= required_ai, "Analog_Input_Requirement"
    prob += provided_ao + provided_uo_analog >= required_ao, "Analog_Output_Requirement"
    
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] == 'Optimal':
        total_cost = pulp.value(prob.objective); solution = {name: int(var.varValue) for name, var in component_qty_vars.items() if var.varValue > 0}
        return total_cost, solution
    else: return None, None
def get_accessories_recursively(parent_components_df, accessories_df):
    full_accessory_list = []; parents_to_check = parent_components_df.copy()
    while not parents_to_check.empty:
        parents_to_check['PartNumber'] = parents_to_check['PartNumber'].str.strip(); accessories_df['ParentPartNumber'] = accessories_df['ParentPartNumber'].str.strip()
        found_accessories = parents_to_check.merge(accessories_df, left_on='PartNumber', right_on='ParentPartNumber', how='inner')
        if found_accessories.empty: break
        found_accessories['Quantity'] = found_accessories['Quantity']
        accessories_for_boq = found_accessories[['AccessoryName', 'AccessoryPartNumber', 'Quantity', 'AccessoryCost']].rename(columns={'AccessoryName': 'Name', 'AccessoryPartNumber': 'PartNumber', 'AccessoryCost': 'Cost'})
        full_accessory_list.append(accessories_for_boq); parents_to_check = accessories_for_boq
    if not full_accessory_list: return pd.DataFrame(columns=['Name', 'PartNumber', 'Quantity', 'Cost'])
    return pd.concat(full_accessory_list)

# --- Authentication Routes ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username]['password'] == password:
            user = User(username)
            login_user(user)
            return redirect(url_for('tool'))
        else:
            flash('Invalid credentials. Please try again.')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# --- Protected Application Routes ---
@app.route('/')
@login_required
def tool():
    return render_template('tool.html')

@app.route('/get_panel_names', methods=['POST'])
@login_required
def get_panel_names():
    file = request.files['panels_file'];
    if not file: return jsonify({"error": "No file uploaded"}), 400
    try:
        panels_df = pd.read_csv(io.BytesIO(file.read())).fillna(0); panels_df['PanelName'] = panels_df['PanelName'].str.strip()
        app.config['PANELS_DF'] = panels_df; return jsonify({"panel_names": panels_df['PanelName'].tolist()})
    except Exception as e: return jsonify({"error": f"Error processing CSV: {str(e)}"}), 500

@app.route('/calculate_options', methods=['POST'])
@login_required
def calculate_options():
    servers = pd.read_csv(os.path.join(BASE_DIR, 'servers.csv')).fillna(0); server_modules = pd.read_csv(os.path.join(BASE_DIR, 'server_modules.csv')).fillna(0)
    accessories = pd.read_csv(os.path.join(BASE_DIR, 'accessories.csv')).fillna(0); data = request.json
    panel_name = data.get('panel_name'); spare_points_percentage = data.get('spare_points_percentage', DEFAULT_SPARE_POINTS_PERCENTAGE)
    panels_df = app.config.get('PANELS_DF')
    if panels_df is None: return jsonify({"error": "Panels data not found. Please upload again."}), 400
    requirements = panels_df[panels_df['PanelName'] == panel_name].iloc[0].to_dict()
    spare_multiplier = 1 + (spare_points_percentage / 100)
    asp_server = servers[servers['Name'].str.contains("AS-P", case=False)].iloc[0]; asb_servers = servers[servers['Name'].str.contains("AS-B", case=False)]
    options = []
    module_cost, modules = find_optimal_combination(requirements, server_modules, spare_points_percentage)
    if modules is not None:
        primary_components = [{'Name': asp_server['Name'], 'PartNumber': asp_server['PartNumber'], 'Quantity': 1, 'Cost': asp_server['Cost']}]
        for name, qty in modules.items():
            info = server_modules[server_modules['Name'] == name].iloc[0]
            primary_components.append({'Name': name, 'PartNumber': info['PartNumber'], 'Quantity': qty, 'Cost': info['Cost']})
        acc_cost = get_accessories_recursively(pd.DataFrame(primary_components), accessories)['Cost'].sum()
        total_cost = asp_server['Cost'] + module_cost + acc_cost
        options.append({'type': 'AS-P', 'name': 'AS-P System', 'cost': total_cost, 'valid': True, 'components': modules})
    for _, asb in asb_servers.iterrows():
        # Apply spare points and round up (no decimals)
        req_inputs=math.ceil((requirements.get('DI',0)+requirements.get('AI',0))*spare_multiplier); req_outputs=math.ceil((requirements.get('DO',0)+requirements.get('AO',0))*spare_multiplier)
        total_req=req_inputs+req_outputs; total_avail=asb.get('DI',0)+asb.get('AI',0)+asb.get('UI',0)+asb.get('DO',0)+asb.get('AO',0)+asb.get('UO',0)+asb.get('UIO',0)
        max_in=asb.get('DI',0)+asb.get('AI',0)+asb.get('UI',0)+asb.get('UIO',0); max_out=asb.get('DO',0)+asb.get('AO',0)+asb.get('UO',0)+asb.get('UIO',0)
        if total_avail>=total_req and max_in>=req_inputs and max_out>=req_outputs:
            acc_cost = get_accessories_recursively(pd.DataFrame([asb]), accessories)['Cost'].sum()
            options.append({'type': 'AS-B', 'name': asb['Name'], 'cost': asb['Cost'] + acc_cost, 'valid': True, 'components': {asb['Name']: 1}})
    return jsonify(sorted(options, key=lambda x: x['cost']))

@app.route('/generate_reports', methods=['POST'])
@login_required
def generate_reports():
    controllers = pd.read_csv(os.path.join(BASE_DIR, 'controllers.csv')).fillna(0); servers = pd.read_csv(os.path.join(BASE_DIR, 'servers.csv')).fillna(0)
    server_modules = pd.read_csv(os.path.join(BASE_DIR, 'server_modules.csv')).fillna(0); accessories = pd.read_csv(os.path.join(BASE_DIR, 'accessories.csv')).fillna(0)
    for df in [controllers, servers, server_modules]: df['Name'] = df['Name'].str.strip()
    data = request.json; panel_choices = data.get('panel_choices'); standard_panels = data.get('standard_panels')
    spare_points_percentage = data.get('spare_points_percentage', DEFAULT_SPARE_POINTS_PERCENTAGE); panels_df = app.config['PANELS_DF']
    all_solutions = []
    for panel_name, choice in panel_choices.items():
        if choice and choice.get('components'):
            for component, qty in choice['components'].items(): all_solutions.append({'PanelName': panel_name, 'ControllerName': component, 'Quantity': qty})
    for panel_name in standard_panels:
        requirements = panels_df[panels_df['PanelName'] == panel_name].iloc[0].to_dict()
        _, result = find_optimal_combination(requirements, controllers, spare_points_percentage)
        if result:
            for component, qty in result.items(): all_solutions.append({'PanelName': panel_name, 'ControllerName': component, 'Quantity': qty})
        else: all_solutions.append({'PanelName': panel_name, 'ControllerName': 'No Solution Found', 'Quantity': 0})
    if not all_solutions: return jsonify({"matrix": [], "boq": []})
    solution_df = pd.DataFrame(all_solutions); pivoted_df = solution_df.pivot_table(index='PanelName', columns='ControllerName', values='Quantity', fill_value=0)
    pivoted_df['SUM'] = pivoted_df.sum(axis=1); pivoted_df = pivoted_df.reset_index()
    cols = pivoted_df.columns.tolist(); main_cols = sorted([c for c in cols if c not in ['PanelName', 'SUM']])
    final_order = ['PanelName'] + main_cols + ['SUM']; pivoted_df = pivoted_df[final_order]
    matrix_json = pivoted_df.to_dict(orient='records')
    all_components = pd.concat([controllers, servers, server_modules]); primary_boq = solution_df[solution_df['ControllerName'] != 'No Solution Found'].groupby('ControllerName')['Quantity'].sum().reset_index()
    if not primary_boq.empty:
        primary_boq = primary_boq.merge(all_components, left_on='ControllerName', right_on='Name', how='left'); accessories_boq = get_accessories_recursively(primary_boq, accessories)
        final_primary_list = primary_boq[['Name', 'PartNumber', 'Quantity', 'Cost']]; final_boq_df = pd.concat([final_primary_list, accessories_boq])
        final_boq_df = final_boq_df.groupby(['Name', 'PartNumber', 'Cost'])['Quantity'].sum().reset_index(); final_boq_df['TotalCost'] = final_boq_df['Quantity'] * final_boq_df['Cost']
        final_boq_df = final_boq_df.rename(columns={'Name': 'ControllerName'}); grand_total = final_boq_df['TotalCost'].sum()
        final_boq_df.loc[len(final_boq_df.index)] = ['Grand Total', '', '', '', grand_total]
        boq_json = final_boq_df.to_dict(orient='records')
    else: boq_json = []
    return jsonify({"matrix": matrix_json, "boq": boq_json})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=False, host='0.0.0.0', port=port)

