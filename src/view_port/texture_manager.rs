use sdl3::{
    gpu::{
        CopyPass, Device, Texture, TextureCreateInfo,
        TextureFormat, TextureRegion, TextureTransferInfo, TextureType,
        TextureUsage, TransferBufferUsage,
    },
    surface::Surface,
    Error,
};

use std::{collections::HashMap, path::Path};

pub struct TexureManager{
    pub texture_table : Vec<Texture<'static>>, //'static prevents de-allocation which could cause problems
    texture_table_lookup : HashMap<String,usize>,
}

impl TexureManager{

    pub fn new() -> Self{
        Self{
            texture_table : Vec::new(),
            texture_table_lookup: HashMap::new(),
        }
    }

//https://github.com/vhspace/sdl3-rs/blob/master/examples/gpu-texture.rs#L425
    pub fn load_texture(
        &mut self,
        gpu: &Device,
        file_path: &str,
        copy_pass: &CopyPass,
        ) -> Result<usize, Error> {

        match  self.texture_table_lookup.get(file_path)
        {
            Some(index) => return Ok(index.clone()),
            None =>{},
        };

        let image = Surface::load_bmp(Path::new(&file_path))?;
        let image_size = image.size();
        let size_bytes = image.pixel_format().bytes_per_pixel() as u32 * image_size.0 * image_size.1;


        self.texture_table.push( gpu.create_texture(
            TextureCreateInfo::new()
                .with_format(TextureFormat::R8g8b8a8Unorm)
                .with_type(TextureType::_2D)
                .with_width(image_size.0)
                .with_height(image_size.1)
                .with_layer_count_or_depth(1)
                .with_num_levels(1)
                .with_usage(TextureUsage::SAMPLER),
        ).unwrap()
        );

        let transfer_buffer = gpu
            .create_transfer_buffer()
            .with_size(size_bytes)
            .with_usage(TransferBufferUsage::UPLOAD)
            .build()?;

        let mut buffer_mem = transfer_buffer.map::<u8>(gpu, false);
        image.with_lock(|image_bytes| {
            buffer_mem.mem_mut().copy_from_slice(image_bytes);
        });
        buffer_mem.unmap();

        copy_pass.upload_to_gpu_texture(
            TextureTransferInfo::new()
                .with_transfer_buffer(&transfer_buffer)
                .with_offset(0),
            TextureRegion::new()
                .with_texture(&self.texture_table.last().unwrap())
                .with_layer(0)
                .with_width(image_size.0)
                .with_height(image_size.1)
                .with_depth(1),
            false,
        );

        drop(transfer_buffer);

        self.texture_table_lookup.insert(file_path.to_string(), self.texture_table.len() - 1);
        Ok(self.texture_table.len() - 1)
    }

    //should work but doesnt
// pub fn get_texture(&self, texture_index : usize
//     ) -> &Texture {

//         return self.texture_table.get(texture_index).unwrap();
//     }

}