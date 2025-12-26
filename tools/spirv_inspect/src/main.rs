use rspirv::binary::Disassemble;
use rspirv::dr::{load_bytes, Module};
use rspirv::spirv::Op;
use std::collections::HashMap;
use std::env;
use std::fs;

fn main() {
    let mut args = env::args().skip(1);

    let path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("Usage: spirv_inspect <path-to-spv> [--store <ptr_id>] [--dump]");
            std::process::exit(2);
        }
    };

    let mut store_ptr_id: Option<u32> = None;
    let mut def_id: Option<u32> = None;
    let mut dump = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--store" => {
                let id_str = args.next().expect("--store requires an id");
                store_ptr_id = Some(id_str.parse::<u32>().expect("invalid id"));
            }
            "--dump" => dump = true,
            "--def" => {
                let id_str = args.next().expect("--def requires an id");
                def_id = Some(id_str.parse::<u32>().expect("invalid id"));
            }
            other => {
                eprintln!("Unknown arg: {other}");
                std::process::exit(2);
            }
        }
    }

    let bytes = fs::read(&path).expect("failed to read spv");
    let module = load_bytes(bytes).expect("failed to parse spv");

    if dump {
        println!("{}", module.disassemble());
        return;
    }

    let analysis = analyze(&module);

    if let Some(id) = def_id {
        println!("Definition for %{id}:");
        if let Some(def) = analysis.def_inst.get(&id) {
            println!("  {def}");
        } else {
            println!("  <no defining instruction found>");
        }
        let name = analysis.names.get(&id).map(|s| s.as_str()).unwrap_or("<unnamed>");
        println!("  name: {name}");
        if let Some(ty) = analysis.type_of_id.get(&id).copied() {
            println!("  type: %{ty}");
            if let Some(def) = analysis.def_inst.get(&ty) {
                println!("  type def: {def}");
            }
            if let Some(pointee) = analysis.pointer_pointee.get(&ty).copied() {
                println!("  pointee type: %{pointee}");
                if let Some(def) = analysis.def_inst.get(&pointee) {
                    println!("  pointee def: {def}");
                }
            }
        }
        return;
    }

    if let Some(ptr_id) = store_ptr_id {
        let mut found = false;
        for (func_name, ptr, obj) in find_opstores(&module, &analysis.names, &analysis.entry_points) {
            if ptr == ptr_id {
                found = true;
                println!("OpStore in function: {func_name}");
                print_store_info(&analysis, ptr, obj);
            }
        }

        if !found {
            eprintln!("No OpStore with ptr id %{ptr_id} found.");
            std::process::exit(1);
        }
    } else {
        // Print all OpStores where the object is a 0 constant.
        for (func_name, ptr, obj) in find_opstores(&module, &analysis.names, &analysis.entry_points) {
            if analysis.const_zero_ids.contains(&obj) {
                println!("OpStore in function: {func_name}");
                print_store_info(&analysis, ptr, obj);
                println!();
            }
        }
    }
}

struct Analysis {
    names: HashMap<u32, String>,
    entry_points: HashMap<u32, String>,
    type_of_id: HashMap<u32, u32>,
    pointer_pointee: HashMap<u32, u32>,
    const_zero_ids: std::collections::HashSet<u32>,
    def_inst: HashMap<u32, String>,
}

fn analyze(module: &Module) -> Analysis {
    let mut names = HashMap::new();
    let mut entry_points = HashMap::new();
    let mut type_of_id = HashMap::new();
    let mut pointer_pointee = HashMap::new();
    let mut const_zero_ids = std::collections::HashSet::new();
    let mut def_inst = HashMap::new();

    for inst in module.all_inst_iter() {
        if let Some(result_id) = inst.result_id {
            def_inst.entry(result_id).or_insert_with(|| inst.disassemble());
        }

        match inst.class.opcode {
            Op::EntryPoint => {
                // operands: execution model, function id, name, interface ids...
                let func_id = inst.operands[1].unwrap_id_ref();
                let name = inst.operands[2].unwrap_literal_string().to_string();
                entry_points.insert(func_id, name);
            }
            Op::Name => {
                let target = inst.operands[0].unwrap_id_ref();
                let name = inst.operands[1].unwrap_literal_string().to_string();
                names.insert(target, name);
            }
            Op::TypePointer => {
                let result_id = inst.result_id.unwrap();
                let pointee = inst.operands[1].unwrap_id_ref();
                pointer_pointee.insert(result_id, pointee);
            }
            Op::Variable => {
                if let (Some(result_type), Some(result_id)) = (inst.result_type, inst.result_id) {
                    type_of_id.insert(result_id, result_type);
                }
            }
            Op::Constant => {
                if let (Some(result_type), Some(result_id)) = (inst.result_type, inst.result_id) {
                    type_of_id.insert(result_id, result_type);
                }

                // Record all integer zero constants.
                let result_id = inst.result_id.unwrap();
                let is_zero = inst
                    .operands
                    .iter()
                    .skip(0)
                    .filter_map(|op| op.literal_int32())
                    .all(|v| v == 0);
                if is_zero {
                    const_zero_ids.insert(result_id);
                }
            }
            _ => {
                if let (Some(result_type), Some(result_id)) = (inst.result_type, inst.result_id) {
                    type_of_id.insert(result_id, result_type);
                }
            }
        }
    }

    Analysis {
        names,
        entry_points,
        type_of_id,
        pointer_pointee,
        const_zero_ids,
        def_inst,
    }
}

fn find_opstores(
    module: &Module,
    names: &HashMap<u32, String>,
    entry_points: &HashMap<u32, String>,
) -> Vec<(String, u32, u32)> {
    let mut out = Vec::new();
    let mut current_fn = String::from("<module>");

    for inst in module.all_inst_iter() {
        match inst.class.opcode {
            Op::Function => {
                current_fn = inst
                    .result_id
                    .map(|id| {
                        if let Some(name) = names.get(&id) {
                            format!("%{id} ({name})")
                        } else if let Some(name) = entry_points.get(&id) {
                            format!("%{id} ({name})")
                        } else {
                            format!("%{id}")
                        }
                    })
                    .unwrap_or_else(|| "<function>".to_string());
            }
            Op::FunctionEnd => current_fn = String::from("<module>"),
            Op::Store => {
                let ptr = inst.operands[0].unwrap_id_ref();
                let obj = inst.operands[1].unwrap_id_ref();
                out.push((current_fn.clone(), ptr, obj));
            }
            _ => {}
        }
    }

    out
}

fn print_store_info(analysis: &Analysis, ptr: u32, obj: u32) {
    let ptr_name = analysis
        .names
        .get(&ptr)
        .map(|s| s.as_str())
        .unwrap_or("<unnamed>");
    let obj_name = analysis
        .names
        .get(&obj)
        .map(|s| s.as_str())
        .unwrap_or("<unnamed>");

    let ptr_type = analysis.type_of_id.get(&ptr).copied();
    let obj_type = analysis.type_of_id.get(&obj).copied();

    println!("  ptr: %{ptr} ({ptr_name})");
    println!("  obj: %{obj} ({obj_name})");
    if let Some(def) = analysis.def_inst.get(&ptr) {
        println!("  ptr def: {def}");
    }
    if let Some(def) = analysis.def_inst.get(&obj) {
        println!("  obj def: {def}");
    }
    println!("  ptr type id: {:?}", ptr_type.map(|t| format!("%{t}")));
    println!("  obj type id: {:?}", obj_type.map(|t| format!("%{t}")));

    if let Some(ptr_type_id) = ptr_type {
        if let Some(def) = analysis.def_inst.get(&ptr_type_id) {
            println!("  ptr type def: {def}");
        }
    }
    if let Some(obj_type_id) = obj_type {
        if let Some(def) = analysis.def_inst.get(&obj_type_id) {
            println!("  obj type def: {def}");
        }
    }

    if let Some(ptr_type_id) = ptr_type {
        if let Some(pointee) = analysis.pointer_pointee.get(&ptr_type_id).copied() {
            let pointee_name = analysis
                .names
                .get(&pointee)
                .map(|s| s.as_str())
                .unwrap_or("<unnamed>");
            println!("  ptr pointee type id: %{pointee} ({pointee_name})");
            if let Some(def) = analysis.def_inst.get(&pointee) {
                println!("  pointee type def: {def}");
            }
        }
    }
}

trait OperandExt {
    fn unwrap_id_ref(&self) -> u32;
    fn unwrap_literal_string(&self) -> &str;
    fn literal_int32(&self) -> Option<u32>;
}

impl OperandExt for rspirv::dr::Operand {
    fn unwrap_id_ref(&self) -> u32 {
        match self {
            rspirv::dr::Operand::IdRef(id) => *id,
            _ => panic!("expected IdRef"),
        }
    }

    fn unwrap_literal_string(&self) -> &str {
        match self {
            rspirv::dr::Operand::LiteralString(s) => s,
            _ => panic!("expected LiteralString"),
        }
    }

    fn literal_int32(&self) -> Option<u32> {
        match self {
            rspirv::dr::Operand::LiteralInt32(v) => Some(*v),
            _ => None,
        }
    }
}
