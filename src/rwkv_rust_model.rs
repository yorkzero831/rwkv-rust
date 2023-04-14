use std::cmp::max;
use std::ffi::CString;
use std::ptr::null_mut;
use crate::{ggml_tensor, ggml_type_GGML_TYPE_F32, rwkv, rwkv_context};

pub struct RkwvModel {
    pub shared_library: rwkv,
    model_path: *const str,
    thread_count: usize,
    pub context: *mut rwkv_context,
    state_buffer_element_count: u32,
    logits_buffer_element_count: u32,
}

impl RkwvModel {
    pub fn new(model_path: &str) -> Self {
        unsafe {
            let lib = rwkv::new("libs/rwkv_avx2.dll").unwrap();
            let c_path = CString::new(model_path).unwrap();
            let context = lib.rwkv_init_from_file(c_path.as_ptr(), 1);
            let state_buffer_element_count = lib.rwkv_get_state_buffer_element_count(context);
            let logits_buffer_element_count = lib.rwkv_get_logits_buffer_element_count(context);
            println!("logits_buffer_element_count: {}", logits_buffer_element_count);
            RkwvModel {
                shared_library: lib,
                model_path,
                thread_count: max(1, num_cpus::get() / 2),
                context,
                state_buffer_element_count,
                logits_buffer_element_count,
            }
        }
    }

    pub fn eval(&self, token: i32, state_in: Option<*mut f32>, state_out: Option<*mut f32>, logits_out: Option<*mut f32>) -> (*mut f32, *mut f32) {
        unsafe {
            let state_in_ptr: *mut f32 = match state_in {
                None => {
                    // let mut vec = vec![0.0f32; usize::try_from(self.state_buffer_element_count).unwrap()];
                    // vec.as_mut_ptr()
                    let out: *mut f32 = null_mut();
                    out

                }
                Some(state_in_v) => {
                    // self.validate_buffer(state_in_v, "state_in", self.state_buffer_element_count);
                    state_in_v
                }
            };

            let state_out_ptr: *mut f32 = match state_out {
                None => {
                    let mut vec = vec![0.0f32; usize::try_from(self.state_buffer_element_count).unwrap()];
                    vec.as_mut_ptr()
                }
                Some(state_out_v) => {
                    //self.validate_buffer(state_in_v, "state_out", self.state_buffer_element_count);
                    state_out_v
                }
            };

            let logits_out_ptr: *mut f32 = match logits_out {
                None => {
                    let mut vec = vec![0.0f32; usize::try_from(self.logits_buffer_element_count).unwrap()];
                    vec.as_mut_ptr()
                }
                Some(logits_out_v) => {
                    //self.validate_buffer(state_in_v, "state_out", self.state_buffer_element_count);
                    logits_out_v
                }
            };

            self.shared_library.rwkv_eval(
                self.context,
                token,
                state_in_ptr,
                state_out_ptr,
                logits_out_ptr,
            );

            (logits_out_ptr, state_out_ptr)
        }
    }

    pub fn free(&self) {
        unsafe {
            self.shared_library.rwkv_free(self.context)
        }
    }
}