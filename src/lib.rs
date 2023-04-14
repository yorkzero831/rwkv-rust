mod rwkv_rust_model;
mod sampling;
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));


#[cfg(test)]
mod tests {
    use std::ffi::CStr;
    use crate::rwkv_rust_model;

    #[test]
    fn it_works() {

        let rwkv_model = rwkv_rust_model::RkwvModel::new("D:/AI/rwkv/RWKV-4-Raven-3B-v7-ChnEng-Q4_1.bin");
        unsafe {
            let system_info =  rwkv_model.shared_library.rwkv_get_system_info_string();
            let system_info_str = CStr::from_ptr(system_info).to_str().unwrap();
            print!("{}", system_info_str)
        }

        // let c_path = CString::new("D:/AI/rwkv/RWKV-4-Raven-3B-v7-ChnEng-Q4_1.bin").unwrap();
        // unsafe {
        //     let rwkv = rwkv::new("libs/rwkv_avx2.dll").unwrap();
        //     let contenxt= rwkv.rwkv_init_from_file(c_path.as_ptr(), 1);
        //     rwkv.rwkv_free(contenxt);
        //     let system_info = rwkv.rwkv_get_system_info_string();
        //     let system_info_str = CStr::from_ptr(system_info).to_str().unwrap();
        //     // let aa = rwkv.rwkv_eval();
        //     print!("{}", system_info_str)
        // }
    }
}
