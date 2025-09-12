from flask import Flask, request, jsonify, render_template
import pandas as pd
import pulp
import io

# Initialize the Flask application
app = Flask(__name__)

# --- Helper Functions ---

def find_optimal_combination(panel_requirements, components_df):
    """
    Solves for the most cost-effective combination of components.
    This is the core optimization engine.
    """
    prob = pulp.LpProblem(f"BMS_Combination_{panel_requirements['PanelName']}", pulp.LpMinimize)
    
    # Use the component's original name directly as the key, which is robust
    component_names = components_df['Name'].tolist()
    component_qty_vars = pulp.LpVariable.dicts("Qty", component_names, lowBound=0, cat='Integer')
    
    # Create a mapping from name to the full component data for easier lookup
    components_map = components_df.set_index('Name').to_dict('index')

    uio_as_input_vars = pulp.LpVariable.dicts("UIO_as_Input", component_names, lowBound=0, cat='Continuous')
    uio_as_output_vars = pulp.LpVariable.dicts("UIO_as_Output", component_names, lowBound=0, cat='Continuous')
    
    prob += pulp.lpSum([components_map[name]['Cost'] * component_qty_vars[name] for name in component_names]), "Total_Component_Cost"

    for name in component_names:
        available_uio = components_map[name].get('UIO', 0) * component_qty_vars[name]
        # Sanitize constraint names for PuLP
        safe_name = ''.join(e for e in name if e.isalnum())
        prob += uio_as_input_vars[name] + uio_as_output_vars[name] <= available_uio, f"UIO_Allocation_{safe_name}"

    total_required_inputs = panel_requirements.get('DI', 0) + panel_requirements.get('AI', 0)
    total_required_outputs = panel_requirements.get('DO', 0) + panel_requirements.get('AO', 0)
    
    total_provided_inputs = pulp.lpSum([(components_map[name].get('DI', 0) + components_map[name].get('AI', 0) + components_map[name].get('UI', 0)) * component_qty_vars[name] for name in component_names]) + pulp.lpSum(uio_as_input_vars)
    total_provided_outputs = pulp.lpSum([(components_map[name].get('DO', 0) + components_map[name].get('AO', 0) + components_map[name].get('UO', 0)) * component_qty_vars[name] for name in component_names]) + pulp.lpSum(uio_as_output_vars)
    
    prob += total_provided_inputs >= total_required_inputs, "Total_Input_Requirement"
    prob += total_provided_outputs >= total_required_outputs, "Total_Output_Requirement"
    
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] == 'Optimal':
        total_cost = pulp.value(prob.objective)
        solution = {name: int(var.varValue) for name, var in component_qty_vars.items() if var.varValue > 0}
        return total_cost, solution
    else:
        return None, None

def get_accessories_recursively(parent_components_df, accessories_df):
    """Finds all accessories, including nested ones."""
    full_accessory_list = []
    parents_to_check = parent_components_df.copy().rename(columns={'Quantity': 'ParentQuantity'})
    
    while not parents_to_check.empty:
        parents_to_check['PartNumber'] = parents_to_check['PartNumber'].astype(str).str.strip()
        accessories_df['ParentPartNumber'] = accessories_df['ParentPartNumber'].astype(str).str.strip()
        
        found_accessories = parents_to_check.merge(accessories_df, left_on='PartNumber', right_on='ParentPartNumber', how='inner')
        if found_accessories.empty: break
        
        found_accessories['Quantity'] = found_accessories['ParentQuantity']
        accessories_for_boq = found_accessories[['AccessoryName', 'AccessoryPartNumber', 'Quantity', 'AccessoryCost']].rename(columns={'AccessoryName': 'Name', 'AccessoryPartNumber': 'PartNumber', 'AccessoryCost': 'Cost'})
        full_accessory_list.append(accessories_for_boq)
        
        parents_to_check = accessories_for_boq.rename(columns={'Quantity': 'ParentQuantity'})

    if not full_accessory_list: return pd.DataFrame(columns=['Name', 'PartNumber', 'Quantity', 'Cost'])
    return pd.concat(full_accessory_list)

# --- Flask API Endpoints ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_panel_names', methods=['POST'])
def get_panel_names():
    file = request.files['panels_file']
    if not file: return jsonify({"error": "No file uploaded"}), 400
    
    try:
        panels_df = pd.read_csv(io.BytesIO(file.read())).fillna(0)
        panels_df['PanelName'] = panels_df['PanelName'].str.strip()
        app.config['PANELS_DF'] = panels_df
        return jsonify({"panel_names": panels_df['PanelName'].tolist()})
    except Exception as e:
        return jsonify({"error": f"Error processing CSV: {str(e)}"}), 500

@app.route('/calculate_options', methods=['POST'])
def calculate_options():
    data = request.json
    panel_name = data.get('panel_name')
    panels_df = app.config.get('PANELS_DF')
    if panels_df is None: return jsonify({"error": "Panels data not found. Please upload again."}), 400
        
    requirements = panels_df[panels_df['PanelName'] == panel_name].iloc[0].to_dict()

    servers = pd.read_csv('servers.csv').fillna(0)
    server_modules = pd.read_csv('server_modules.csv').fillna(0)
    accessories = pd.read_csv('accessories.csv').fillna(0)
    
    asp_server = servers[servers['Name'].str.contains("AS-P", case=False)].iloc[0]
    asb_servers = servers[servers['Name'].str.contains("AS-B", case=False)]
    options = []
    
    module_cost, modules = find_optimal_combination(requirements, server_modules)
    if modules is not None:
        primary_components = [{'Name': asp_server['Name'], 'PartNumber': asp_server['PartNumber'], 'Quantity': 1, 'Cost': asp_server['Cost']}]
        for name, qty in modules.items():
            info = server_modules[server_modules['Name'] == name].iloc[0]
            primary_components.append({'Name': name, 'PartNumber': info['PartNumber'], 'Quantity': qty, 'Cost': info['Cost']})
        acc_cost = get_accessories_recursively(pd.DataFrame(primary_components), accessories)['Cost'].sum()
        total_cost = asp_server['Cost'] + module_cost + acc_cost
        options.append({'type': 'AS-P', 'name': 'AS-P System', 'cost': total_cost, 'valid': True, 'components': modules})
    
    for _, asb in asb_servers.iterrows():
        req_inputs=requirements.get('DI',0)+requirements.get('AI',0); req_outputs=requirements.get('DO',0)+requirements.get('AO',0)
        total_req=req_inputs+req_outputs
        total_avail=asb.get('DI',0)+asb.get('AI',0)+asb.get('UI',0)+asb.get('DO',0)+asb.get('AO',0)+asb.get('UO',0)+asb.get('UIO',0)
        max_in=asb.get('DI',0)+asb.get('AI',0)+asb.get('UI',0)+asb.get('UIO',0)
        max_out=asb.get('DO',0)+asb.get('AO',0)+asb.get('UO',0)+asb.get('UIO',0)
        if total_avail>=total_req and max_in>=req_inputs and max_out>=req_outputs:
            acc_cost = get_accessories_recursively(pd.DataFrame([asb]), accessories)['Cost'].sum()
            options.append({'type': 'AS-B', 'name': asb['Name'], 'cost': asb['Cost'] + acc_cost, 'valid': True, 'components': {asb['Name']: 1}})

    return jsonify(sorted(options, key=lambda x: x['cost']))

@app.route('/generate_reports', methods=['POST'])
def generate_reports():
    data = request.json
    panel_choices = data.get('panel_choices')
    standard_panels = data.get('standard_panels')
    panels_df = app.config['PANELS_DF']
    
    controllers=pd.read_csv('controllers.csv').fillna(0); servers=pd.read_csv('servers.csv').fillna(0); server_modules=pd.read_csv('server_modules.csv').fillna(0); accessories=pd.read_csv('accessories.csv').fillna(0)
    for df in [controllers, servers, server_modules]: df['Name'] = df['Name'].str.strip()

    all_solutions = []
    for panel_name, choice in panel_choices.items():
        if choice and choice.get('components'):
            for component, qty in choice['components'].items(): all_solutions.append({'PanelName': panel_name, 'ControllerName': component, 'Quantity': qty})
    for panel_name in standard_panels:
        requirements = panels_df[panels_df['PanelName'] == panel_name].iloc[0].to_dict()
        _, result = find_optimal_combination(requirements, controllers)
        if result:
            for component, qty in result.items(): all_solutions.append({'PanelName': panel_name, 'ControllerName': component, 'Quantity': qty})
        else: all_solutions.append({'PanelName': panel_name, 'ControllerName': 'No Solution Found', 'Quantity': 0})
    
    if not all_solutions: return jsonify({"matrix": [], "boq": []})

    solution_df = pd.DataFrame(all_solutions)

    # --- Generate Matrix with Correct Column Order ---
    pivoted_df = solution_df.pivot_table(index='PanelName', columns='ControllerName', values='Quantity', fill_value=0)
    pivoted_df['SUM'] = pivoted_df.sum(axis=1)
    pivoted_df = pivoted_df.reset_index()
    
    # **FIX**: Explicitly set the column order
    cols = pivoted_df.columns.tolist()
    main_cols = sorted([c for c in cols if c not in ['PanelName', 'SUM']])
    final_order = ['PanelName'] + main_cols + ['SUM']
    pivoted_df = pivoted_df[final_order]
    
    matrix_json = pivoted_df.to_dict(orient='records')
    
    # --- Generate BOQ ---
    all_components = pd.concat([controllers, servers, server_modules])
    primary_boq = solution_df[solution_df['ControllerName'] != 'No Solution Found'].groupby('ControllerName')['Quantity'].sum().reset_index()
    if not primary_boq.empty:
        primary_boq = primary_boq.merge(all_components, left_on='ControllerName', right_on='Name', how='left')
        accessories_boq = get_accessories_recursively(primary_boq, accessories)
        final_primary_list = primary_boq[['Name', 'PartNumber', 'Quantity', 'Cost']]
        final_boq_df = pd.concat([final_primary_list, accessories_boq])
        final_boq_df = final_boq_df.groupby(['Name', 'PartNumber', 'Cost'])['Quantity'].sum().reset_index()
        final_boq_df['TotalCost'] = final_boq_df['Quantity'] * final_boq_df['Cost']
        final_boq_df = final_boq_df.rename(columns={'Name': 'ControllerName'})
        grand_total = final_boq_df['TotalCost'].sum()
        final_boq_df.loc[len(final_boq_df.index)] = ['Grand Total', '', '', '', grand_total]
        boq_json = final_boq_df.to_dict(orient='records')
    else: boq_json = []

    return jsonify({"matrix": matrix_json, "boq": boq_json})

if __name__ == '__main__':
    app.run(debug=True, port=5001)

