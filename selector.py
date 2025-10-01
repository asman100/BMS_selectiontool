# bms_selector_final_robust.py

import pandas as pd
import pulp
import math

# Default spare points percentage (can be overridden by user input)
DEFAULT_SPARE_POINTS_PERCENTAGE = 20

def find_optimal_combination(panel_requirements, components_df, spare_points_percentage=None):
    """
    Solves for the most cost-effective combination of components (controllers or modules).
    """
    if spare_points_percentage is None:
        spare_points_percentage = DEFAULT_SPARE_POINTS_PERCENTAGE
    spare_multiplier = 1 + (spare_points_percentage / 100)
    
    prob = pulp.LpProblem(f"BMS_Combination_{panel_requirements.name}", pulp.LpMinimize)
    
    # Use the component name directly as the key, which is robust
    component_names = components_df['Name'].tolist()
    component_qty_vars = pulp.LpVariable.dicts("Qty", component_names, lowBound=0, cat='Integer')
    
    # Create a mapping from name to the full component data for easier lookup
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
        # UIO constraint: total UIO usage across all four types cannot exceed available
        available_uio = components_map[name]['UIO'] * component_qty_vars[name]
        prob += uio_as_di_vars[name] + uio_as_ai_vars[name] + uio_as_do_vars[name] + uio_as_ao_vars[name] <= available_uio, f"UIO_Allocation_{name.replace(' ', '_')}"
        # UI constraint: can be used for digital OR analog input (not both)
        available_ui = components_map[name]['UI'] * component_qty_vars[name]
        prob += ui_as_digital_vars[name] + ui_as_analog_vars[name] <= available_ui, f"UI_Allocation_{name.replace(' ', '_')}"
        # UO constraint: can be used for digital OR analog output (not both)
        available_uo = components_map[name]['UO'] * component_qty_vars[name]
        prob += uo_as_digital_vars[name] + uo_as_analog_vars[name] <= available_uo, f"UO_Allocation_{name.replace(' ', '_')}"

    # Apply spare points and round up (no decimals)
    required_di = math.ceil(panel_requirements['DI'] * spare_multiplier)
    required_do = math.ceil(panel_requirements['DO'] * spare_multiplier)
    required_ai = math.ceil(panel_requirements['AI'] * spare_multiplier)
    required_ao = math.ceil(panel_requirements['AO'] * spare_multiplier)
    
    # Individual constraints for each point type (Digital and Analog separately)
    provided_di = pulp.lpSum([components_map[name]['DI'] * component_qty_vars[name] for name in component_names])
    provided_do = pulp.lpSum([components_map[name]['DO'] * component_qty_vars[name] for name in component_names])
    provided_ai = pulp.lpSum([components_map[name]['AI'] * component_qty_vars[name] for name in component_names])
    provided_ao = pulp.lpSum([components_map[name]['AO'] * component_qty_vars[name] for name in component_names])
    
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
        total_cost = pulp.value(prob.objective)
        # --- *** CRITICAL FIX *** ---
        # Extract results using the original names from the variable dictionary keys,
        # which avoids any errors from name sanitization by PuLP.
        solution = {}
        for name, var in component_qty_vars.items():
            if var.varValue > 0:
                solution[name] = int(var.varValue)
        # --- END OF FIX ---
        return total_cost, solution
    else:
        return None, None

def get_accessories_recursively(parent_components_df, accessories_df):
    # This function is unchanged...
    full_accessory_list = []
    parents_to_check = parent_components_df.copy()
    while not parents_to_check.empty:
        # Ensure keys are stripped of whitespace for a clean merge
        parents_to_check['PartNumber'] = parents_to_check['PartNumber'].str.strip()
        accessories_df['ParentPartNumber'] = accessories_df['ParentPartNumber'].str.strip()
        
        found_accessories = parents_to_check.merge(accessories_df, left_on='PartNumber', right_on='ParentPartNumber', how='inner')
        if found_accessories.empty: break
        
        found_accessories['Quantity'] = found_accessories['Quantity']
        accessories_for_boq = found_accessories[['AccessoryName', 'AccessoryPartNumber', 'Quantity', 'AccessoryCost']].rename(columns={'AccessoryName': 'Name', 'AccessoryPartNumber': 'PartNumber', 'AccessoryCost': 'Cost'})
        full_accessory_list.append(accessories_for_boq)
        parents_to_check = accessories_for_boq
    if not full_accessory_list: return pd.DataFrame(columns=['Name', 'PartNumber', 'Quantity', 'Cost'])
    return pd.concat(full_accessory_list)

# --- Main Script Execution ---
if __name__ == "__main__":
    try:
        # Load all data files and strip whitespace from name columns to prevent merge errors
        controllers = pd.read_csv('controllers.csv').fillna(0)
        controllers['Name'] = controllers['Name'].str.strip()
        servers = pd.read_csv('servers.csv').fillna(0)
        servers['Name'] = servers['Name'].str.strip()
        server_modules = pd.read_csv('server_modules.csv').fillna(0)
        server_modules['Name'] = server_modules['Name'].str.strip()
        panels = pd.read_csv('panels.csv')
        panels['PanelName'] = panels['PanelName'].str.strip()
        accessories = pd.read_csv('accessories.csv').fillna(0)
        print("✅ Successfully loaded all CSV files.")

        # --- Interactive Prompts ---
        project_name_raw = input("\nEnter the Project Name, then press Enter:\n> ")
        project_name = project_name_raw.strip().replace(' ', '_')
        print(f"--> Project set to: {project_name}\n")
        
        spare_points_input = input("Enter Spare Points Percentage (default 20%, press Enter for default):\n> ")
        spare_points_percentage = int(spare_points_input) if spare_points_input.strip() else DEFAULT_SPARE_POINTS_PERCENTAGE
        print(f"--> Spare points set to: {spare_points_percentage}%\n")
        
        panel_names = panels['PanelName'].tolist()
        print("Available Panels:\n" + ", ".join(panel_names))
        user_input = input("\nEnter the names of panels to be Automation Servers (comma-separated):\n> ")
        server_panel_names = {name.strip() for name in user_input.split(',') if name.strip()}
        
        # --- Interactive Decision Loop ---
        panel_server_choices = {}
        asp_server = servers[servers['Name'].str.contains("AS-P", case=False)].iloc[0]
        asb_servers = servers[servers['Name'].str.contains("AS-B", case=False)]
        spare_multiplier = 1 + (spare_points_percentage / 100)
        for panel_name in sorted(list(server_panel_names)):
            print("-" * 40)
            print(f"DECISION for Server Panel: {panel_name}")
            requirements = panels[panels['PanelName'] == panel_name].iloc[0]
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
            else:
                options.append({'type': 'AS-P', 'name': 'AS-P System', 'cost': float('inf'), 'valid': False})
            for index, asb in asb_servers.iterrows():
                # Apply spare points and round up (no decimals)
                req_inputs = math.ceil((requirements['DI'] + requirements['AI']) * spare_multiplier); req_outputs = math.ceil((requirements['DO'] + requirements['AO']) * spare_multiplier)
                total_required_points = req_inputs + req_outputs
                total_available_points = asb['DI'] + asb['AI'] + asb['UI'] + asb['DO'] + asb['AO'] + asb['UO'] + asb['UIO']
                max_possible_inputs = asb['DI'] + asb['AI'] + asb['UI'] + asb['UIO']
                max_possible_outputs = asb['DO'] + asb['AO'] + asb['UO'] + asb['UIO']
                is_valid = (total_available_points >= total_required_points) and (max_possible_inputs >= req_inputs) and (max_possible_outputs >= req_outputs)
                if is_valid:
                    acc_cost = get_accessories_recursively(pd.DataFrame([asb]), accessories)['Cost'].sum()
                    total_cost = asb['Cost'] + acc_cost
                    options.append({'type': 'AS-B', 'name': asb['Name'], 'cost': total_cost, 'valid': True, 'components': {asb['Name']: 1}})
                else:
                    options.append({'type': 'AS-B', 'name': asb['Name'], 'cost': float('inf'), 'valid': False})
            print("Please choose an option for this panel:")
            valid_choices = []
            for i, option in enumerate(options):
                if option['valid']: print(f"  [{i+1}] {option['name']}: Total Cost = ${option['cost']:.2f}"); valid_choices.append(str(i+1))
                else: print(f"  [x] {option['name']}: Not a valid option (I/O requirements not met)")
            if not valid_choices:
                print("  No valid server options found for this panel's requirements."); panel_server_choices[panel_name] = {'No Valid Server Solution': 0}; continue
            choice = '';
            while choice not in valid_choices: choice = input(f"Enter your choice for {panel_name} ({', '.join(valid_choices)}): ").strip()
            panel_server_choices[panel_name] = options[int(choice)-1]['components']
            if options[int(choice)-1]['type'] == 'AS-P': panel_server_choices[panel_name][asp_server['Name']] = 1
        
        # --- Main Solution-Building Loop ---
        all_solutions = []
        for index, row in panels.iterrows():
            panel_name = row['PanelName']
            if panel_name in panel_server_choices:
                for component, qty in panel_server_choices[panel_name].items():
                    all_solutions.append({'PanelName': panel_name, 'ControllerName': component.strip(), 'Quantity': qty})
            elif panel_name not in server_panel_names:
                _, result = find_optimal_combination(row, controllers, spare_points_percentage)
                if result:
                    for component, qty in result.items():
                        all_solutions.append({'PanelName': panel_name, 'ControllerName': component.strip(), 'Quantity': qty})
                else:
                    all_solutions.append({'PanelName': panel_name, 'ControllerName': 'No Solution Found', 'Quantity': 0})
        
        # --- Generate and Save Outputs (Unchanged from previous correct version) ---
        if all_solutions:
            solution_df = pd.DataFrame(all_solutions)
            print("\nGenerating Pivoted Panel Matrix...")
            pivoted_df = solution_df.pivot_table(index='PanelName', columns='ControllerName', values='Quantity', fill_value=0)
            if 'SUM' not in pivoted_df.columns: pivoted_df['SUM'] = pivoted_df.sum(axis=1)
            pivoted_df.to_csv(f"{project_name}_panel_matrix_solution.csv")
            print(f"📄 Pivoted matrix saved to '{project_name}_panel_matrix_solution.csv'")
            print("\nGenerating Project Bill of Quantities (BOQ) with Accessories...")
            all_components = pd.concat([controllers, servers, server_modules])
            primary_components_boq = solution_df.groupby('ControllerName')['Quantity'].sum().reset_index()
            primary_components_boq = primary_components_boq.merge(all_components,left_on='ControllerName',right_on='Name',how='left')
            accessories_boq = get_accessories_recursively(primary_components_boq, accessories)
            final_primary_list = primary_components_boq[['Name', 'PartNumber', 'Quantity', 'Cost']]
            final_boq_df = pd.concat([final_primary_list, accessories_boq])
            final_boq_df = final_boq_df.groupby(['Name', 'PartNumber', 'Cost'])['Quantity'].sum().reset_index()
            final_boq_df['TotalCost'] = final_boq_df['Quantity'] * final_boq_df['Cost']
            final_boq_df = final_boq_df.rename(columns={'Name': 'ControllerName'})
            grand_total = final_boq_df['TotalCost'].sum()
            final_boq_df.loc[len(final_boq_df.index)] = ['Grand Total', '', '', '', grand_total]
            final_boq_df.to_csv(f"{project_name}_project_boq.csv", index=False)
            print(f"🧾 Project BOQ with accessories saved to '{project_name}_project_boq.csv'")
            print(f"\n🎉 All panels processed successfully!")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure all five CSV files are in the same directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")