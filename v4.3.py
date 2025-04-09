import streamlit as st
import pandas as pd
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Tuple, Dict
import plotly.express as px
import copy

# Define data structure for warehouse equipment
@dataclass
class WarehouseEquipment:
    equipment_id: str
    equipment_type: str
    brand: str
    model: str
    cost: float
    sq_ft: float
    priority: int
    description: str

def calculate_type_summary(selected: List[WarehouseEquipment]) -> dict:
    """Calculate summary statistics for each equipment type, including total quantity."""
    summary = defaultdict(lambda: {
        'total_qty': 0,
        'total_cost': 0.0,
        'total_sq_ft': 0.0
    })

    for item in selected:
        eq_type = item.equipment_type
        # Accumulate total quantity
        summary[eq_type]['total_qty']  += item.quantity
        # Accumulate total cost & space for all units
        summary[eq_type]['total_cost'] += item.cost * item.quantity
        summary[eq_type]['total_sq_ft'] += item.sq_ft * item.quantity

    return dict(summary)


def estimate_storage_capacity(selected: List[WarehouseEquipment]) -> float:
    """Estimate storage capacity based on equipment types and space."""
    # Basic estimation - can be refined based on requirements
    return sum(item.sq_ft * 1.5 for item in selected if item.equipment_type == "Storage System")  # e.g., 1.5 units per sq ft

def plan_warehouse_with_budget_allocation(
    budget: float, 
    total_sq_ft: float, 
    equipment_list: List[WarehouseEquipment],
    budget_allocations: Dict[str, float]
) -> Tuple[List[WarehouseEquipment], float, float]:
    """
    Round-robin approach:
      - Flatten & sort by priority (desc).
      - In each pass, try to buy exactly 1 unit of each item if budget/space allows.
      - If we purchase anything during that pass, repeat.
      - Stop when a full pass yields no purchases.
      - Accumulate quantities in the final result.
    """

    # 1) Copy the equipment list so we don't modify the original
    equipment_list_copy = copy.deepcopy(equipment_list)

    # Group by type so we can track each type’s remaining allocated budget
    equipment_by_type = defaultdict(list)
    for eq in equipment_list_copy:
        equipment_by_type[eq.equipment_type].append(eq)

    for eq_type in equipment_by_type:
        equipment_by_type[eq_type].sort(key=lambda x: x.priority, reverse=True)

    # Convert allocation % into actual dollar amounts
    type_budgets = {
        eq_type: budget * (pct / 100.0) for eq_type, pct in budget_allocations.items()
    }

    remaining_budget = budget
    remaining_space = total_sq_ft

    # Flatten all equipment into a single list (still sorted by priority)
    # so we can do a pass from highest to lowest
    flattened_list = []
    for eq_type, items in equipment_by_type.items():
        flattened_list.extend(items)
    flattened_list.sort(key=lambda x: x.priority, reverse=True)

    # We'll accumulate purchases in a dict keyed by (eq_type, eq_id)
    # The value will hold a reference to the equipment (for display fields) and a `quantity`.
    selected_dict: Dict[Tuple[str, str], dict] = {}

    # 2) Round-robin passes
    purchased_something = True
    while purchased_something:
        purchased_something = False

        # Single pass over the entire flattened list
        for item in flattened_list:
            eq_type = item.equipment_type
            if eq_type not in type_budgets:
                # If eq_type not in allocations, treat as 0 budget or skip
                continue

            # Try buying 1 unit if feasible
            if (
                item.cost <= type_budgets[eq_type]  # fits this type's remaining allocation
                and item.cost <= remaining_budget
                and item.sq_ft <= remaining_space
            ):
                # Purchase 1 unit, increment the quantity
                key = (eq_type, item.equipment_id)
                if key not in selected_dict:
                    # Clone the item so each has its own 'quantity' tracking
                    selected_dict[key] = {
                        "equipment": copy.deepcopy(item),
                        "quantity": 0
                    }
                selected_dict[key]["quantity"] += 1

                # Deduct from budgets/space
                type_budgets[eq_type] -= item.cost
                remaining_budget     -= item.cost
                remaining_space     -= item.sq_ft

                purchased_something = True

        # If a full pass completes with no purchases, we're done.

    # 3) Build final list of WarehouseEquipment, each with the correct quantity.
    #    We'll keep cost/sq_ft as single-unit values, so the total can be item.cost * quantity when displayed.
    selected_equipment: List[WarehouseEquipment] = []
    for (eq_type, eq_id), data in selected_dict.items():
        eq_obj = data["equipment"]
        qty    = data["quantity"]

        # eq_obj already holds the single-unit cost and sq_ft
        # Update eq_obj.quantity so it's displayed properly
        eq_obj.quantity = qty

        selected_equipment.append(eq_obj)

    return selected_equipment, remaining_budget, remaining_space



def plan_warehouse(budget: float, total_sq_ft: float, equipment_list: List[WarehouseEquipment]) -> Tuple[List[WarehouseEquipment], float, float]:
    """Legacy optimize warehouse equipment selection based on constraints."""
    # Sort equipment by priority (highest first)
    sorted_equipment = sorted(equipment_list, key=lambda x: x.priority, reverse=True)
    
    selected_equipment = []
    remaining_budget = budget
    remaining_space = total_sq_ft
    
    # Select equipment that fits within constraints
    for equipment in sorted_equipment:
        if equipment.cost <= remaining_budget and equipment.sq_ft <= remaining_space:
            selected_equipment.append(equipment)
            remaining_budget -= equipment.cost
            remaining_space -= equipment.sq_ft
    
    return selected_equipment, remaining_budget, remaining_space

def main():
    st.title("Warehouse Equipment Planning Tool")
    
    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    # Left column for numerical inputs
    with col1:
        st.subheader("Enter Warehouse Parameters")
        budget = st.number_input("Budget ($)", min_value=0.0, value=150000.0, step=1000.0)
        total_sq_ft = st.selectbox("Total Square Footage", [20000,25000,30000])

    # Right column for file upload
    with col2:
        st.subheader("Upload Equipment Data")
        st.write("Upload Excel file with required columns")
        uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx'])
        
        # DISPLAY EXAMPLE FORMAT
        st.write("Expected columns:")
        example_df = pd.DataFrame({
            'Equipment ID': ['EQ001', 'EQ002'],
            'Equipment Type': ['Forklift', 'Pallet Rack'],
            'Brand': ['Toyota', 'Uline'],
            'Model': ['8FBE20', 'H-4636'],
            'Unit Cost': [35000, 25000],
            'Square Footage': [100, 500],
            'Priority Score': [5, 4],
            'Description': ['Electric Forklift', 'Steel Pallet Rack']
        })
        st.dataframe(example_df)

    # Advanced recommendation feature
    st.subheader("Recommendation Settings")
    use_advanced = st.checkbox("Advanced Recommendation")
    
    # Initialize budget allocations dictionary
    budget_allocations = {}
    
    # Default categories
    categories = [
        "Material Handling Equipment",
        "Storage System",
        "Dock Equipment",
        "Shipping Supplies / Equipment",
        "Workspace Equipment",
        "Miscellaneous"
    ]
    categories_lower = [c.lower() for c in categories]
    
    # Display budget allocation sliders if advanced recommendation is checked
    if use_advanced:
        tab1, tab2 = st.tabs(["Budget Allocation", "Floor Space Utilization"])
        
        with tab1:
            st.write("Allocate budget percentages to each equipment category:")
            st.write("Note: Total allocation must equal 100%")
            
            
            

            # Initialize session state for sliders if not exists
            if 'slider_values' not in st.session_state:
                # Start with equal distribution
                initial_value = 100 / len(categories)
                st.session_state.slider_values = {category: initial_value for category in categories}
                st.session_state.prev_adjusted = None
            
            # Function to adjust other sliders when one changes
            def adjust_other_sliders(changed_category, new_value):
                old_value = st.session_state.slider_values[changed_category]
                delta = new_value - old_value
                
                # If no change or trivial change, do nothing
                if abs(delta) < 0.1:
                    return
                    
                # Update the changed slider value
                st.session_state.slider_values[changed_category] = new_value
                
                # Calculate how much to distribute to other sliders
                other_categories = [c for c in categories if c != changed_category]
                
                # Skip the previously adjusted slider if possible to avoid ping-pong
                if st.session_state.prev_adjusted in other_categories and len(other_categories) > 1:
                    other_categories.remove(st.session_state.prev_adjusted)
                
                # Calculate the total of other sliders
                other_total = sum(st.session_state.slider_values[c] for c in other_categories)
                
                # If other total is too small, adjust all sliders
                if other_total < abs(delta) or other_total < 0.1:
                    # Reset distribution
                    remaining = 100 - new_value
                    per_category = remaining / len(other_categories) if other_categories else 0
                    for c in other_categories:
                        st.session_state.slider_values[c] = per_category
                else:
                    # Distribute the change proportionally
                    for c in other_categories:
                        proportion = st.session_state.slider_values[c] / other_total
                        adjustment = -delta * proportion
                        st.session_state.slider_values[c] += adjustment
                
                # Ensure we end up with exactly 100%
                current_total = sum(st.session_state.slider_values.values())
                if abs(current_total - 100) > 0.1:
                    # Fix rounding errors
                    adjust = 100 - current_total
                    # Distribute to the first category with a non-zero value
                    for c in other_categories:
                        if st.session_state.slider_values[c] > 0.1:
                            st.session_state.slider_values[c] += adjust
                            break
                
                # Remember which category was adjusted
                st.session_state.prev_adjusted = changed_category
            
            # Create sliders
            for category in categories:
                # Create a unique key for each slider
                slider_key = f"slider_{category}"
                
                # Get the current value from session state
                current_value = st.session_state.slider_values[category]
                
                # Create the slider
                new_value = st.slider(
                    f"{category}", 
                    min_value=0.0, 
                    max_value=100.0, 
                    value=float(current_value),
                    step=1.0,
                    format="%.0f%%",
                    key=slider_key,
                    help=f"Percentage of budget allocated to {category}"
                )
                
                # Check if the value changed
                if new_value != current_value:
                    adjust_other_sliders(category, new_value)
                
                # Update the budget allocations dictionary
                budget_allocations[category] = st.session_state.slider_values[category]
            
            # Show the current total
            total_allocation = sum(budget_allocations.values())
            st.write(f"Total allocation: {total_allocation:.0f}%")
            
            # Warning if not 100%
            if abs(total_allocation - 100) > 0.1:
                st.warning("⚠️ Total allocation must equal 100%. Please adjust your allocations.")
            #Budget Allocation Block End *************************************************************************************


        #Floor Space Utilization Slider Block
        with tab2:
            floor_space_utilization = st.slider(
            "Floor space utilization",
            min_value=0.0,
            max_value=100.0,
            value=100.0,   # default to 100%
            step=1.0,
            format="%.0f%%",
            help="Use only this percentage of the total warehouse square footage for planning."
            )
    
            # Calculate the effective square footage based on that slider
            effective_sq_ft = total_sq_ft * (floor_space_utilization / 100.0)
            st.write(f"Effective Warehouse Space for Planning: **{effective_sq_ft:.0f} sq ft**")
        #Floor Space Utilization Slider Block End

    if uploaded_file is not None:
        try:
            # Process uploaded file
            df = pd.read_excel(uploaded_file)
            required_columns = [
                'Equipment ID', 'Equipment Type', 'Brand', 'Model',
                'Unit Cost', 'Square Footage', 'Priority Score', 'Description'
            ]
            
            # Validate columns
            if not all(col in df.columns for col in required_columns):
                st.error("Excel file must contain all required columns: " + ", ".join(required_columns))
                return
            
            # DISPLAY UPLOADED DATA
            st.subheader("Uploaded Equipment Data")
            with st.expander("📂 Uploaded Equipment Data", expanded=False):
                st.dataframe(df)
            
            # Get unique equipment types from the data
            unique_equipment_types = df['Equipment Type'].unique()
            
            # Ensure our default categories match what's in the data
            missing_types = [t for t in unique_equipment_types if t.lower() not in categories_lower]
            if missing_types:
                st.warning(f"Notice: The following equipment types from your data aren't in the predefined categories: {', '.join(missing_types)}")
                st.info("These items will be treated as 'Miscellaneous' in advanced recommendations.")
            
            
            # Standardize uploaded equipment type casing to match predefined categories
            def standardize_type(equip_type):
                for cat in categories:
                    if equip_type.lower() == cat.lower():
                        return cat  # Return the official casing
                return equip_type  # Leave as-is if no match found

            df['Equipment Type'] = df['Equipment Type'].apply(standardize_type)

            # Convert DataFrame to equipment list
            equipment_list = [
                WarehouseEquipment(
                    equipment_id=str(row['Equipment ID']),
                    equipment_type=str(row['Equipment Type']),
                    brand=str(row['Brand']),
                    model=str(row['Model']),
                    cost=float(row['Unit Cost']),
                    sq_ft=float(row['Square Footage']),
                    priority=int(row['Priority Score']),
                    description=str(row['Description'])
                )
                for _, row in df.iterrows()
            ]
            
            # Generate recommendations when button is clicked
            if st.button("Generate Recommendations"):
                # Check if we can proceed with advanced recommendations
                can_proceed = True
                if use_advanced and abs(sum(budget_allocations.values()) - 100) > 0.1:
                    st.error("Cannot proceed: Budget allocations must sum to 100%")
                    can_proceed = False
                
                if can_proceed:
                    # Select planning function based on advanced setting
                    if use_advanced:
                        selected, remaining_budget, remaining_space = plan_warehouse_with_budget_allocation(
                            budget, effective_sq_ft, equipment_list, budget_allocations
                        )
                    else:
                        selected, remaining_budget, remaining_space = plan_warehouse(
                            budget, effective_sq_ft, equipment_list
                        )
                    
                    # 1. Display the individual equipment recommendations, grouped by type
                    st.header("Individual Equipment Recommendations")
                    # Sort selected items by equipment_type so they appear grouped
                    selected_sorted = sorted(selected, key=lambda x: x.equipment_type)
                    
                    recommendations_df = pd.DataFrame({
                        'Equipment Type': [item.equipment_type for item in selected_sorted],
                        'Equipment ID':   [item.equipment_id for item in selected_sorted],
                        'Brand':          [item.brand for item in selected_sorted],
                        'Model':          [item.model for item in selected_sorted],
                        # Single-unit cost and sq_ft:
                        'Unit Cost ($)':  [item.cost for item in selected_sorted],
                        'Unit SqFt':      [item.sq_ft for item in selected_sorted],
                        # Show quantity, total cost, total sq_ft:
                        'Quantity':       [item.quantity for item in selected_sorted],
                        'Total Cost ($)': [
                            round(item.cost * item.quantity, 2) for item in selected_sorted
                        ],
                        'Total SqFt': [
                            round(item.sq_ft * item.quantity, 2) for item in selected_sorted
                         ],
                        'Priority Score': [item.priority for item in selected_sorted],
                        'Description':    [item.description for item in selected_sorted]
})

                    st.dataframe(recommendations_df)
                    
                    # 2. Display a summary by equipment type
                    st.header("Equipment Type Summary")
                    type_summary = calculate_type_summary(selected)
                    summary_data = []
                    for eq_type, stats in type_summary.items():
                        total_qty = stats['total_qty']
                        total_cost = stats['total_cost']
                        total_sq_ft = stats['total_sq_ft']
    
                        # If you'd like to provide an Average Unit Cost across the type
                        avg_unit_cost = 0
                        if total_qty > 0:
                            avg_unit_cost = round(total_cost / total_qty, 2)
    
                        summary_data.append({
                            'Equipment Type': eq_type,
                            'Quantity': total_qty,
                            'Avg Unit Cost': avg_unit_cost,
                            'Total Cost': round(total_cost, 2),
                            'Square Footage': round(total_sq_ft, 2)
                        })
                    
                    # Add budget allocation percentages if using advanced mode
                    if use_advanced:
                        for item in summary_data:
                            eq_type = item['Equipment Type']
                            if eq_type in budget_allocations:
                                item['Budget Allocation'] = f"{budget_allocations[eq_type]:.0f}%"
                                item['Allocated Budget'] = budget * (budget_allocations[eq_type] / 100)
                            else:
                                # If the type wasn't in our predefined categories, it uses Miscellaneous allocation
                                item['Budget Allocation'] = f"{budget_allocations['Miscellaneous']:.0f}%"
                                item['Allocated Budget'] = budget * (budget_allocations['Miscellaneous'] / 100)
                    
                    st.dataframe(pd.DataFrame(summary_data))
                    
                    # SPATIAL ANALYSIS
                    st.header("Spatial Analysis")
                    used_space = sum(item.sq_ft for item in selected)
                    storage_capacity = estimate_storage_capacity(selected)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Square Footage", f"{total_sq_ft:,.0f} sq ft")
                    with col2:
                        st.metric("Equipment Footprint", f"{used_space:,.0f} sq ft")
                    with col3:
                        st.metric("Remaining Space", f"{remaining_space:,.0f} sq ft")
                    with col4:
                        st.metric("Storage Capacity Estimate", f"{storage_capacity:,.0f} units")
                    
                    # SPACE UTILIZATION CHART
                    st.subheader("Space Utilization")
                    space_data = pd.DataFrame({
                        'Category': ['Used Space', 'Remaining Space'],
                        'Square Footage': [used_space, remaining_space]
                    })

                    space_fig = px.pie(
                        space_data, 
                        values='Square Footage', 
                        names='Category', 
                        color = 'Category',
                        color_discrete_map={
                            'Used Space': 'red',
                            'Remaining Space': 'green'
                        }
                    )
                    st.plotly_chart(space_fig)
                    
                    # FINANCIAL SUMMARY
                    st.header("Financial Summary")
                    total_cost = sum(item.cost for item in selected)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Initial Budget", f"${budget:,.2f}")
                    with col2:
                        st.metric("Total Equipment Cost", f"${total_cost:,.2f}")
                    with col3:
                        st.metric("Remaining Budget", f"${remaining_budget:,.2f}")
                    
                    # BUDGET UTILIZATION CHART
                    st.subheader("Budget Utilization")
                    budget_data = pd.DataFrame({
                        'Category': ['Equipment Cost', 'Remaining Budget'],
                        'Amount': [total_cost, remaining_budget]
                    })

                    budget_fig = px.pie(
                        budget_data, 
                        values='Amount', 
                        names='Category', 
                        color = 'Category',
                        color_discrete_map={
                            'Equipment Cost': 'red',
                            'Remaining Budget': 'green'
                        }
                    )
                    st.plotly_chart(budget_fig)
                    
                    # If advanced recommendation was used, display budget allocation breakdown
                    if use_advanced:
                        st.subheader("Budget Allocation by Equipment Type")
                        
                        # Create dataframe for budget allocations
                        allocation_data = []
                        for eq_type, allocation in budget_allocations.items():
                            allocated_amount = budget * (allocation / 100)
                            # Find how much was actually spent on this type
                            spent = sum(item.cost for item in selected if item.equipment_type == eq_type)
                            
                            allocation_data.append({
                                'Equipment Type': eq_type,
                                'Budget Allocation (%)': f"{allocation:.0f}%",
                                'Allocated Amount ($)': allocated_amount,
                                'Spent Amount ($)': spent,
                                'Remaining ($)': allocated_amount - spent
                            })
                        
                        st.dataframe(pd.DataFrame(allocation_data))
                        
                        # Budget allocation chart
                        st.subheader("Budget Allocation Chart")
                        alloc_chart_data = pd.DataFrame({
                            'Equipment Type': [item['Equipment Type'] for item in allocation_data],
                            'Allocation': [item['Allocated Amount ($)'] for item in allocation_data]
                        })
                        
                        alloc_fig = px.bar(
                            alloc_chart_data,
                            x='Equipment Type',
                            y='Allocation',
                            title='Budget Allocation by Equipment Type',
                            labels={'Allocation': 'Budget Allocation ($)'}
                        )
                        st.plotly_chart(alloc_fig)
                    
                    # DOWNLOAD BUTTONS
                    st.subheader("Download Reports")
                    # Type Summary CSV
                    csv = pd.DataFrame(summary_data).to_csv(index=False)
                    st.download_button(
                        label="Download Type Summary as CSV",
                        data=csv,
                        file_name="warehouse_recommendations_summary.csv",
                        mime="text/csv"
                    )
                    
                    # Individual recommendations CSV
                    csv_recs = recommendations_df.to_csv(index=False)
                    st.download_button(
                        label="Download Individual Recommendations as CSV",
                        data=csv_recs,
                        file_name="warehouse_recommendations_individual.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()

    