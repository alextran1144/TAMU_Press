import streamlit as st
import pandas as pd
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Tuple, Dict
import plotly.express as px
import copy
import math

# ----------------------------------------------------------------
# 1) Data Structure
# ----------------------------------------------------------------
@dataclass
class WarehouseEquipment:
    equipment_id: str
    equipment_type: str    # e.g. "Material Handling Equipment", "Storage System"
    subcategory: str       # e.g. "Forklift", "Pallet Rack"
    brand: str
    model: str
    cost: float
    sq_ft: float
    priority: int
    description: str

# ----------------------------------------------------------------
# 2) Minimum Requirements
# ----------------------------------------------------------------
def calculate_storage_requirements(total_sq_ft: float, 
                                   storage_items: List[WarehouseEquipment],
                                   aisle_width: float = 12.5) -> Tuple[int, WarehouseEquipment]:
    """
    Calculate how many racks can fit in the storage area (70% of total) with specified aisle width.
    Returns the number of racks needed and the best rack choice.
    """
    if not storage_items:
        return 0, None
    
    # Find the highest priority rack
    storage_items.sort(key=lambda x: x.priority, reverse=True)
    best_rack = storage_items[0]
    
    # Calculate how many racks fit in the storage area (70% of total)
    storage_area = total_sq_ft * 0.7
    
    # Calculate the area needed per rack including aisle space
    side = math.sqrt(best_rack.sq_ft)  # Approximate width of rack
    aisle_area = aisle_width * side     # Area used by aisles
    area_per_rack = best_rack.sq_ft + aisle_area
    
    # Calculate maximum number of racks that fit
    max_racks = int(storage_area // area_per_rack)
    
    return max_racks, best_rack

#------------------------------------------
#Space Usage function
#__________________________________________

def calculate_space_usage(selected, total_sq_ft):
    """
    Calculate space usage with proper handling of the storage area.
    When storage is fully utilized, it should equal exactly 70% of total warehouse space.
    """
    # Storage allocation is hardcoded at 70%
    storage_allocation = 0.7
    max_storage_area = total_sq_ft * storage_allocation
    
    # Separate storage and non-storage items
    storage_items = [item for item in selected if item.equipment_type.strip().lower() == "storage system"]
    non_storage_items = [item for item in selected if item.equipment_type.strip().lower() != "storage system"]
    
    # Calculate non-storage space usage
    non_storage_space = 0
    for item in non_storage_items:
        # Parse quantity
        quantity = 1
        if "(Quantity:" in item.description:
            start = item.description.find("(Quantity:") + len("(Quantity:")
            end = item.description.find(")", start)
            if end > start:
                try:
                    quantity = int(item.description[start:end].strip())
                except:
                    quantity = 1
        
        # If the item already includes quantity in its sq_ft, use as is
        if "_x" in item.equipment_id:
            non_storage_space += item.sq_ft
        else:
            non_storage_space += (item.sq_ft * quantity)
    
    # Calculate storage space usage
    if not storage_items:
        storage_space = 0
    else:
        # Check if we've hit the maximum racks that can fit
        needed_racks = 0
        actual_racks = 0
        
        # Calculate the theoretical maximum number of racks
        for item in storage_items:
            # Get the rack area per unit
            sq_ft_per_unit = item.sq_ft
            quantity = 1
            
            if "(Quantity:" in item.description:
                start = item.description.find("(Quantity:") + len("(Quantity:")
                end = item.description.find(")", start)
                if end > start:
                    try:
                        quantity = int(item.description[start:end].strip())
                        sq_ft_per_unit = item.sq_ft / quantity
                    except:
                        pass
                        
            # Calculate required aisle space per rack
            side = math.sqrt(sq_ft_per_unit)
            aisle_area = 12.5 * side
            area_per_rack = sq_ft_per_unit + aisle_area
            
            # Calculate maximum possible racks in the storage area
            theoretical_max_racks = int(max_storage_area // area_per_rack)
            needed_racks = theoretical_max_racks
            
            # Count actual racks
            actual_racks += quantity
        
        # If we've maxed out the racks (or close enough), use the full allocation
        # Allow for 90% utilization to count as "maxed out" to handle rounding issues
        if actual_racks >= needed_racks * 0.9:
            storage_space = max_storage_area
        else:
            # Otherwise calculate the actual space used including aisles
            storage_space = 0
            for item in storage_items:
                quantity = 1
                if "(Quantity:" in item.description:
                    start = item.description.find("(Quantity:") + len("(Quantity:")
                    end = item.description.find(")", start)
                    if end > start:
                        try:
                            quantity = int(item.description[start:end].strip())
                        except:
                            quantity = 1
                
                # Calculate both the rack area and the aisle area
                sq_ft_per_unit = item.sq_ft / quantity if quantity > 1 else item.sq_ft
                side = math.sqrt(sq_ft_per_unit)
                aisle_area = 12.5 * side
                
                # Total space = rack space + aisle space
                storage_space += (sq_ft_per_unit * quantity) + (aisle_area * quantity)
    
    # Total space used
    total_used = storage_space + non_storage_space
    
    # Ensure we don't exceed total warehouse space due to rounding errors
    total_used = min(total_used, total_sq_ft)
    
    return {
        'total_space': total_sq_ft,
        'storage_space': storage_space,
        'non_storage_space': non_storage_space,
        'total_used': total_used,
        'remaining_space': total_sq_ft - total_used
    }
# ----------------------------------------------------------------
# 3) Checklist Function (Enhanced)
# ----------------------------------------------------------------
def generate_checklist(
    selected: List[WarehouseEquipment], 
    eq_list: List[WarehouseEquipment], 
    total_sq_ft: float
) -> pd.DataFrame:
    """Updated checklist that handles quantities in item descriptions."""
    rows = []
    
    # Get minimum requirements for this warehouse size
    requirements = get_requirements(total_sq_ft)
    
    # Parse quantities from selected items
    selected_counts = defaultdict(int)
    for item in selected:
        eq_type = item.equipment_type.strip()
        subcat = item.subcategory.strip()
        
        # Check for quantity in the description
        quantity = 1
        if "(Quantity:" in item.description:
            start = item.description.find("(Quantity:") + len("(Quantity:")
            end = item.description.find(")", start)
            if end > start:
                try:
                    quantity = int(item.description[start:end].strip())
                except:
                    quantity = 1
        
        selected_counts[(eq_type, subcat)] += quantity
    


    
    # First, handle Storage System
    storage_items = [x for x in eq_list if x.equipment_type.strip() == "Storage System"]
    if storage_items:
        needed_racks, _ = calculate_storage_requirements(total_sq_ft, storage_items)
        actual_racks = selected_counts.get(("Storage System", "Rack"), 0)
        if actual_racks == 0:
            # Try with less specific matching
            for key, count in selected_counts.items():
                if key[0] == "Storage System":
                    actual_racks += count
        
        met_str = "Yes" if actual_racks >= needed_racks else "No"
        rows.append({
            "Equipment Type": "Storage System",
            "Required": f"{needed_racks} racks (70% + 12.5 ft aisle)",
            "Selected Count": actual_racks,
            "Met?": met_str
        })
    
    # Process other equipment types
    for eq_type, subcats in requirements.items():
        if eq_type == "Storage System":
            continue  # Already handled above
        
        for subcat, qty_needed in subcats.items():
            actual_qty = selected_counts.get((eq_type, subcat), 0)
            
            # Try with less specific matching if exact match fails
            if actual_qty == 0:
                # Check for partial matches
                for key, count in selected_counts.items():
                    if key[0] == eq_type and subcat.lower() in key[1].lower():
                        actual_qty += count
            
            met_str = "Yes" if actual_qty >= qty_needed else "No"
            rows.append({
                "Equipment Type": eq_type,
                "Required": f"{qty_needed} x {subcat}",
                "Selected Count": actual_qty,
                "Met?": met_str
            })
    
    return pd.DataFrame(rows)


# ----------------------------------------------------------------
# 4) Estimate Storage Capacity
# ----------------------------------------------------------------
def estimate_storage_capacity(selected: List[WarehouseEquipment]) -> int:
    """
    Estimate how many standard 48"x40" pallets can be stored
    based on the footprints of the selected 'Storage System' items.
    
    We'll assume each item.sq_ft represents the usable area for pallets.
    Standard pallet area = 48 in × 40 in = 1920 in² = 1920/144 = 13.33 ft² (approx).
    """
    # Each pallet occupies ~13.33 square feet
    standard_pallet_area = (48.0 * 40.0) / 144.0  # ~13.33 ft²
    
    total_pallet_slots = 0.0
    for item in selected:
        if item.equipment_type.strip().lower() == "storage system":
            # Add how many pallets fit into this storage item's footprint
            total_pallet_slots += (item.sq_ft / standard_pallet_area)

    # Return an integer count of pallets (floor or round as you prefer)
    return int(total_pallet_slots)

# ----------------------------------------------------------------
# 5) Budget-allocation plan with quantity-based Storage System
# ----------------------------------------------------------------
def get_requirements(warehouse_size: int) -> Dict[str, Dict[str, int]]:
    """Get the minimum equipment requirements based on warehouse size."""
    base_requirements = {
        "Storage System": {
            "Rack": 0,  # Will be calculated based on space
        },
        "Material Handling Equipment": {
            "Forklift": 1,
            "Pallet Jack": 2,
        },
        "Dock Equipment": {
            "Dock Seal": 1,
            "Dock Leveler": 1,
            "Guard Rail": 2,
        },
        "Packing and Sorting Systems": {
            "Packing Station": 1,
            "Stretch Wrap Machine": 1,
            "Steel Shelving": 1,
        }
    }
    
    # Adjust requirements based on warehouse size
    if warehouse_size == 25000:
        base_requirements["Material Handling Equipment"]["Roller System"] = 5
        base_requirements["Material Handling Equipment"]["Pallet Jack"] = 3
    elif warehouse_size == 30000:
        base_requirements["Material Handling Equipment"]["Pallet Jack"] = 4
        base_requirements["Material Handling Equipment"]["Roller System"] = 10
    
    return base_requirements
#Helper function to help select optimal equipment set:

def plan_warehouse_with_budget_allocation(
    budget: float, 
    total_sq_ft: float, 
    equipment_list: List[WarehouseEquipment],
    budget_allocations: Dict[str, float]
) -> Tuple[List[WarehouseEquipment], float, float]:
    """
    Enhanced version that handles multiple quantities for minimum requirements.
    Miscellaneous items are handled automatically with remaining budget.
    """
    eq_copy = copy.deepcopy(equipment_list)
    
    # Get minimum requirements based on warehouse size
    min_requirements = get_requirements(total_sq_ft)
    
    # Separate equipment by type
    storage_items = [x for x in eq_copy if x.equipment_type == "Storage System"]
    misc_items = [x for x in eq_copy if x.equipment_type == "Miscellaneous" or 
                 x.equipment_type not in budget_allocations]
    core_items = [x for x in eq_copy if x.equipment_type != "Storage System" and 
                 x.equipment_type != "Miscellaneous" and 
                 x.equipment_type in budget_allocations]
    
    # Convert allocations to dollar amounts (distribute 100% among core categories)
    total_allocation_pct = sum(budget_allocations.values())
    type_budgets = {}
    for eq_type, pct in budget_allocations.items():
        # Scale to ensure we're using 100% of budget among core categories
        scaled_pct = (pct / total_allocation_pct) * 100.0
        type_budgets[eq_type] = budget * (scaled_pct / 100.0)
    
    selected = []
    remaining_budget = budget
    remaining_space = total_sq_ft
    
    # --- A) Storage System logic with quantity
    if storage_items:
        storage_items.sort(key=lambda x: x.priority, reverse=True)
        top_rack = storage_items[0]
        
        side = math.sqrt(top_rack.sq_ft)
        aisle_area = 12.5 * side
        area_per_rack = top_rack.sq_ft + aisle_area
        allowed_area = total_sq_ft * 0.70
        max_by_space = int(allowed_area // area_per_rack)
        
        storage_budget = type_budgets.get("Storage System", 0)
        
        # Calculate maximum racks by storage budget
        max_by_storage_budget = int(storage_budget // top_rack.cost)
        
        # Determine how many we need to buy with allocated budget
        racks_to_buy = min(max_by_space, max_by_storage_budget)
        
        # If we can't buy enough racks with allocated budget, use remaining budget
        if racks_to_buy < max_by_space:
            # Calculate how many more racks we can buy with remaining budget
            additional_racks_needed = max_by_space - racks_to_buy
            additional_budget_needed = additional_racks_needed * top_rack.cost
            
            # Only use additional budget if there's enough
            if additional_budget_needed <= remaining_budget:
                racks_to_buy = max_by_space
        
        if racks_to_buy > 0:
            total_cost = top_rack.cost * racks_to_buy
            total_area = top_rack.sq_ft * racks_to_buy
            aggregated_item = WarehouseEquipment(
                equipment_id=f"{top_rack.equipment_id}_x{racks_to_buy}",
                equipment_type="Storage System",
                subcategory=top_rack.subcategory,
                brand=top_rack.brand,
                model=top_rack.model,
                cost=total_cost,
                sq_ft=total_area,
                priority=top_rack.priority,
                description=f"{top_rack.description} (Quantity: {racks_to_buy})"
            )
            selected.append(aggregated_item)
            
            # Deduct from storage budget first, then from remaining budget if needed
            if total_cost <= storage_budget:
                type_budgets["Storage System"] -= total_cost
            else:
                used_from_storage = storage_budget
                used_from_remaining = total_cost - used_from_storage
                type_budgets["Storage System"] = 0  # Used all storage budget
                # The remaining was already included in the total budget
            
            remaining_budget -= total_cost
            remaining_space -= total_area
    
    # --- B) Core categories - with minimum requirements handling
    # Create a list of unfulfilled requirements to try again with remaining budget
    unfulfilled_requirements = []
    
    # Group equipment by type and subcategory
    equipment_by_type = defaultdict(lambda: defaultdict(list))
    for item in core_items:
        equipment_by_type[item.equipment_type][item.subcategory].append(item)
    
    # First pass: Meet minimum requirements using category budgets
    for eq_type, subcats in min_requirements.items():
        if eq_type == "Storage System":
            continue  # Already handled above
        
        type_budget = type_budgets.get(eq_type, 0)
        
        for subcat, qty_needed in subcats.items():
            if qty_needed <= 0:
                continue
            
            if subcat in equipment_by_type[eq_type] and equipment_by_type[eq_type][subcat]:
                # Get best item of this subcategory
                best_items = sorted(equipment_by_type[eq_type][subcat], key=lambda x: x.priority, reverse=True)
                best_item = best_items[0]
                
                # Calculate total cost and space for required quantity
                total_cost = best_item.cost * qty_needed
                total_space = best_item.sq_ft * qty_needed
                
                # Check if we can afford it with category budget
                if total_cost <= type_budget and total_cost <= remaining_budget and total_space <= remaining_space:
                    # Add with quantity
                    if qty_needed > 1:
                        # Create aggregated item
                        aggregated_item = WarehouseEquipment(
                            equipment_id=f"{best_item.equipment_id}_x{qty_needed}",
                            equipment_type=best_item.equipment_type,
                            subcategory=best_item.subcategory,
                            brand=best_item.brand,
                            model=best_item.model,
                            cost=total_cost,
                            sq_ft=total_space,
                            priority=best_item.priority,
                            description=f"{best_item.description} (Quantity: {qty_needed})"
                        )
                        selected.append(aggregated_item)
                    else:
                        # Just add the single item
                        selected.append(best_item)
                    
                    # Update budgets and space
                    remaining_budget -= total_cost
                    remaining_space -= total_space
                    type_budget -= total_cost
                    type_budgets[eq_type] = type_budget
                else:
                    # Can't afford with category budget, save for second pass
                    unfulfilled_requirements.append((eq_type, subcat, qty_needed, best_item))
    
    # Second pass: Try to meet unfulfilled requirements with remaining budget
    for eq_type, subcat, qty_needed, best_item in unfulfilled_requirements:
        total_cost = best_item.cost * qty_needed
        total_space = best_item.sq_ft * qty_needed
        
        if total_cost <= remaining_budget and total_space <= remaining_space:
            # Add with quantity
            if qty_needed > 1:
                # Create aggregated item
                aggregated_item = WarehouseEquipment(
                    equipment_id=f"{best_item.equipment_id}_x{qty_needed}",
                    equipment_type=best_item.equipment_type,
                    subcategory=best_item.subcategory,
                    brand=best_item.brand,
                    model=best_item.model,
                    cost=total_cost,
                    sq_ft=total_space,
                    priority=best_item.priority,
                    description=f"{best_item.description} (Quantity: {qty_needed})"
                )
                selected.append(aggregated_item)
            else:
                # Just add the single item
                selected.append(best_item)
            
            # Update only remaining budget and space
            remaining_budget -= total_cost
            remaining_space -= total_space
    
    # Third pass: Use remaining category budgets for additional core items
    remaining_core_items = []
    for eq_type, subcats in equipment_by_type.items():
        for subcat, items in subcats.items():
            # Skip items that were already added for minimum requirements
            already_added_subcats = set()
            for item in selected:
                if item.equipment_type == eq_type:
                    already_added_subcats.add(item.subcategory)
            
            if subcat in already_added_subcats:
                continue
            
            remaining_core_items.extend(items)
    
    # Sort by priority and add what fits within category budgets
    remaining_core_items.sort(key=lambda x: x.priority, reverse=True)
    for item in remaining_core_items:
        type_budget = type_budgets.get(item.equipment_type, 0)
        if item.cost <= type_budget and item.cost <= remaining_budget and item.sq_ft <= remaining_space:
            selected.append(item)
            remaining_budget -= item.cost
            remaining_space -= item.sq_ft
            type_budgets[item.equipment_type] -= item.cost
    
    # --- C) Miscellaneous items with remaining budget
    # Sort miscellaneous items by priority
    misc_items.sort(key=lambda x: x.priority, reverse=True)
    
    # Add miscellaneous items until budget or space runs out
    for item in misc_items:
        if item.cost <= remaining_budget and item.sq_ft <= remaining_space:
            selected.append(item)
            remaining_budget -= item.cost
            remaining_space -= item.sq_ft
    
    return selected, remaining_budget, remaining_space

# ----------------------------------------------------------------
# 6) Streamlit UI (Only Budget Allocation)
# ----------------------------------------------------------------
def main():
    st.title("Warehouse Equipment Planning Tool (Budget Allocation Only)")

    # Inputs
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Enter Warehouse Parameters")
        budget = st.number_input("Budget ($)", min_value=0.0, value=150000.0, step=1000.0)
        total_sq_ft = st.selectbox("Total Square Footage", [20000, 25000, 30000])

    with col2:
        st.subheader("Upload Excel file")
        st.write("Required columns:")
        example_df = pd.DataFrame({
            'Equipment ID': ['EQ001','EQ002'],
            'Equipment Type': ['Material Handling Equipment','Dock Equipment'],
            'Subcategory': ['Forklift','Dock Leveler'],
            'Brand': ['Toyota','BlueGiant'],
            'Model': ['8FBE20','DG-123'],
            'Unit Cost': [35000,12000],
            'Square Footage': [100, 50],
            'Priority Score': [5,4],
            'Description': ['Electric Forklift','Basic Dock Leveler']
        })
        st.dataframe(example_df)
        uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx'])

    # We always do budget allocations now
    categories = [
        "Material Handling Equipment",
        "Storage System",
        "Dock Equipment",
        "Packing and Sorting Systems"
    ]
    budget_allocations = {}

    st.subheader("Budget Allocation (Must total 100%)")
    if 'slider_values' not in st.session_state:
        default_val = 100.0 / len(categories)
        st.session_state.slider_values = {cat: default_val for cat in categories}
        st.session_state.prev_adjusted = None

    def adjust_other_sliders(changed, newval):
        oldval = st.session_state.slider_values[changed]
        delta = newval - oldval
        if abs(delta) < 0.1:
            return
        st.session_state.slider_values[changed] = newval
        others = [c for c in categories if c != changed]
        if st.session_state.prev_adjusted in others and len(others) > 1:
            others.remove(st.session_state.prev_adjusted)
        other_total = sum(st.session_state.slider_values[o] for o in others)
        if other_total < abs(delta) or other_total < 0.1:
            remain = 100 - newval
            if others:
                each = remain / len(others)
                for o in others:
                    st.session_state.slider_values[o] = each
        else:
            for o in others:
                prop = st.session_state.slider_values[o] / other_total
                st.session_state.slider_values[o] += (-delta * prop)

        tot_now = sum(st.session_state.slider_values.values())
        if abs(tot_now - 100) > 0.1:
            fix_amount = 100 - tot_now
            for o in others:
                if st.session_state.slider_values[o] > 0.1:
                    st.session_state.slider_values[o] += fix_amount
                    break
        st.session_state.prev_adjusted = changed

    total_pct = 0.0
    for cat in categories:
        cur_val = st.session_state.slider_values[cat]
        val = st.slider(
            cat, 0.0, 100.0, float(cur_val), 1.0,
            key=f"slider_{cat}"
        )
        if val != cur_val:
            adjust_other_sliders(cat, val)
        budget_allocations[cat] = st.session_state.slider_values[cat]
        total_pct += st.session_state.slider_values[cat]

    st.write(f"Total: {total_pct:.0f}%")
    if abs(total_pct - 100) > 1e-6:
        st.warning("Allocations must total 100%")

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            required_cols = [
                "Equipment ID","Equipment Type","Subcategory","Brand","Model",
                "Unit Cost","Square Footage","Priority Score","Description"
            ]
            if not all(col in df.columns for col in required_cols):
                st.error(f"Excel must have columns: {', '.join(required_cols)}")
                return

            st.subheader("Uploaded Equipment Data")
            st.dataframe(df)

            # Convert DataFrame to list
            eq_list = []
            for _, row in df.iterrows():
                eq_list.append(
                    WarehouseEquipment(
                        equipment_id=str(row["Equipment ID"]),
                        equipment_type=str(row["Equipment Type"]),
                        subcategory=str(row["Subcategory"]) if not pd.isna(row["Subcategory"]) else "",
                        brand=str(row["Brand"]),
                        model=str(row["Model"]),
                        cost=float(row["Unit Cost"]),
                        sq_ft=float(row["Square Footage"]),
                        priority=int(row["Priority Score"]),
                        description=str(row["Description"])
                    )
                )

            if st.button("Generate Recommendations"):
                if abs(sum(budget_allocations.values()) - 100) > 1e-6:
                    st.error("Cannot proceed: Budget allocations must total 100%")
                else:
                    selected, rem_budget, rem_space = plan_warehouse_with_budget_allocation(
                        budget, total_sq_ft, eq_list, budget_allocations
                    )

                    # Utility to parse quantity from "(Quantity: X)" in the description
                    def parse_quantity(desc: str) -> int:
                        if "(Quantity:" in desc:
                            start = desc.find("(Quantity:") + len("(Quantity:")
                            end = desc.find(")", start)
                            if end > start:
                                return int(desc[start:end].strip())
                        return 1

                    # Build DataFrame of selected items
                    rows_list = []
                    for item in selected:
                        q = parse_quantity(item.description)
                        rows_list.append({
                            "Equipment Type": item.equipment_type,
                            "Subcategory": item.subcategory,
                            "Equipment ID": item.equipment_id,
                            "Brand": item.brand,
                            "Model": item.model,
                            "Unit Cost": item.cost,
                            "Square Footage": item.sq_ft,
                            "Priority": item.priority,
                            "Quantity": q,
                            "Description": item.description
                        })
                    recs_df = pd.DataFrame(rows_list)

                    st.header("Individual Equipment Recommendations")
                    st.dataframe(recs_df)

                    # Checklist (enhanced storage row)
                    st.header("Minimum Requirements Checklist")
                    checklist_df = generate_checklist(selected, eq_list, total_sq_ft)
                    st.dataframe(checklist_df)

                    # Summary by Equipment Type
                    st.header("Equipment Type Summary")
                    summary_rows = []
                    grouped = recs_df.groupby("Equipment Type")
                    for eq_type, group in grouped:
                        total_q = group["Quantity"].sum()
                        total_cost = group["Unit Cost"].sum()
                        total_area = group["Square Footage"].sum()
                        avg_cost = total_cost / len(group) if len(group) else 0
                        summary_rows.append({
                            "Equipment Type": eq_type,
                            "Quantity Sum": total_q,
                            "Total Cost": round(total_cost, 2),
                            "Total Sq Ft": round(total_area, 2),
                            "Avg Unit Cost": round(avg_cost, 2)
                        })
                    summary_df = pd.DataFrame(summary_rows)
                    st.dataframe(summary_df)

                    # 1) A helper to parse quantity out of the "(Quantity: X)" in the description
                    def parse_quantity(desc: str) -> int:
                        if "(Quantity:" in desc:
                            start = desc.find("(Quantity:") + len("(Quantity:")
                            end = desc.find(")", start)
                            if end > start:
                                try:
                                    return int(desc[start:end].strip())
                                except:
                                    pass
                        return 1

                    # Spatial Analysis
                    st.header("Spatial Analysis")

                    # Calculate space usage with improved logic
                    space_usage = calculate_space_usage(selected, total_sq_ft)

                    cA, cB, cC, cD = st.columns(4)
                    with cA:
                        st.metric("Total Sq Ft", f"{space_usage['total_space']:,.0f}")
                    with cB:
                        st.metric("Used Space", f"{space_usage['total_used']:,.0f}")
                    with cC:
                        st.metric("Remaining Space", f"{space_usage['remaining_space']:,.0f}")
                    with cD:
                        storage_cap = estimate_storage_capacity(selected)
                        st.metric("Storage Capacity Est.", f"{storage_cap:,.0f} Pallets")

                    # Provide detailed space breakdown
                    st.subheader("Space Breakdown")
                    st.write(f"Storage Area: {space_usage['storage_space']:,.0f} sq ft ({space_usage['storage_space']/total_sq_ft:.1%} of total)")
                

                    # Pie chart for used vs. remaining
                    space_data = pd.DataFrame({
                        "Category": ["Used Space", "Remaining Space"],
                        "SqFt": [space_usage['total_used'], space_usage['remaining_space']]
                    })
                    fig_space = px.pie(space_data, values="SqFt", names="Category")
                    st.plotly_chart(fig_space)
                    # Budget Allocation usage
                    st.subheader("Budget Allocation by Equipment Type")
                    alloc_rows = []
                    for cat in budget_allocations:
                        allocated_amt = budget * (budget_allocations[cat]/100)
                        spent_cat = sum(it.cost for it in selected if it.equipment_type == cat)
                        alloc_rows.append({
                            "Equipment Type": cat,
                            "Allocation(%)": f"{budget_allocations[cat]:.0f}%",
                            "Allocated($)": allocated_amt,
                            "Spent($)": spent_cat,
                            "Remaining($)": allocated_amt - spent_cat
                        })
                    alloc_df = pd.DataFrame(alloc_rows)
                    st.dataframe(alloc_df)

                    st.subheader("Allocation Chart")
                    chart_data = pd.DataFrame({
                        "Equipment Type": [r["Equipment Type"] for r in alloc_rows],
                        "Allocation": [r["Allocated($)"] for r in alloc_rows]
                    })
                    fig_alloc = px.bar(chart_data, x="Equipment Type", y="Allocation")
                    st.plotly_chart(fig_alloc)

                    # Download Buttons
                    st.subheader("Download Reports")
                    sum_csv = summary_df.to_csv(index=False)
                    st.download_button(
                        label="Download Type Summary (CSV)",
                        data=sum_csv,
                        file_name="type_summary.csv",
                        mime="text/csv"
                    )
                    recs_csv = recs_df.to_csv(index=False)
                    st.download_button(
                        label="Download Individual Recs (CSV)",
                        data=recs_csv,
                        file_name="recommendations.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
