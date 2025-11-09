use base64::{Engine as _, engine::general_purpose};
use clap::{Arg, Command};
use indicatif::{ProgressBar, ProgressStyle};
use log::{error, warn};
use reqwest::Client;
use serde_json::{Value, json};
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

fn encode_image_to_base64(path: &Path) -> Result<(String, String), String> {
    let mut file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)
        .map_err(|e| format!("Failed to read file: {}", e))?;

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let mime_type = match ext.as_str() {
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "bmp" => "image/bmp",
        "gif" => "image/gif",
        "webp" => "image/webp",
        _ => return Err(format!("Unsupported image extension: .{}", ext)),
    };

    let b64_data = general_purpose::STANDARD.encode(&data);
    Ok((b64_data, mime_type.to_string()))
}

async fn call_ollama_structured(
    client: &Client,
    api_url: &str,
    model: &str,
    image_b64: &str,
    prompt: &str,
    schema_obj: Option<&Value>,
    options: Option<&Value>,
) -> Result<Value, String> {
    let messages = vec![json!({
        "role": "user",
        "content": prompt,
        "images": [image_b64],
    })];

    let mut payload = json!({
        "model": model,
        "messages": messages,
        "stream": false,
    });

    if let Some(schema) = schema_obj {
        payload["format"] = schema.clone();
    }
    if let Some(opts) = options {
        payload["options"] = opts.clone();
    }

    let resp = client
        .post(api_url)
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .await
        .map_err(|e| format!("HTTP request failed: {}", e))?;

    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();

    if !status.is_success() {
        error!("Server said: {}", text);
        return Err(format!("HTTP error: {}", status));
    }

    serde_json::from_str(&text).map_err(|e| format!("Failed to parse JSON: {}", e))
}

#[tokio::main]
async fn main() {
    let matches = Command::new("Ollama Batch Image Captioning")
        .arg(
            Arg::new("dir")
                .long("dir")
                .required(true)
                .help("Directory of input images"),
        )
        .arg(
            Arg::new("api_url")
                .long("api-url")
                .required(true)
                .help("Ollama API URL"),
        )
        .arg(
            Arg::new("model")
                .long("model")
                .required(true)
                .help("Model name in Ollama"),
        )
        .arg(
            Arg::new("schema")
                .long("schema")
                .help("JSON schema file path (optional)"),
        )
        .arg(
            Arg::new("prompt")
                .long("prompt")
                .default_value("What do you see in this image?")
                .help("Prompt to send to the model"),
        )
        .arg(
            Arg::new("output_dir")
                .long("output-dir")
                .help("Directory to save output JSON files"),
        )
        .arg(
            Arg::new("debug")
                .long("debug")
                .action(clap::ArgAction::SetTrue)
                .help("Enable verbose debug logging"),
        )
        .arg(
            Arg::new("options")
                .long("options")
                .help("JSON string of additional model options"),
        )
        .arg(
            Arg::new("pretty_json")
                .long("pretty-json")
                .action(clap::ArgAction::SetTrue)
                .help("Pretty format the JSON"),
        )
        .arg(
            Arg::new("batch_size")
                .long("batch-size")
                .default_value("1")
                .help("Number of images per batch"),
        )
        .arg(
            Arg::new("skip_existing")
                .long("skip-existing")
                .action(clap::ArgAction::SetTrue)
                .help("Skip any images which already have JSON for them."),
        )
        .arg(
            Arg::new("suffix")
                .long("suffix")
                .default_value("")
                .help("Suffix to append to JSON file names."),
        )
        .get_matches();

    let input_dir = matches.get_one::<String>("dir").unwrap();
    let api_url = matches.get_one::<String>("api_url").unwrap();
    let model = matches.get_one::<String>("model").unwrap();
    let prompt = matches.get_one::<String>("prompt").unwrap();
    let output_dir = matches
        .get_one::<String>("output_dir")
        .map(|s| s.as_str())
        .unwrap_or(input_dir);
    let pretty_json = matches.get_flag("pretty_json");
    let batch_size: usize = matches
        .get_one::<String>("batch_size")
        .unwrap()
        .parse()
        .unwrap_or(1);

    let skip_existing = matches.get_flag("skip_existing");
    let file_suffix = matches.get_one::<String>("suffix").unwrap();

    let schema_obj = matches.get_one::<String>("schema").map(|schema_path| {
        let schema_str = fs::read_to_string(schema_path).expect("Failed to read schema file");
        serde_json::from_str(&schema_str).expect("Invalid JSON schema")
    });

    let options = matches.get_one::<String>("options").map(|opts| {
        serde_json::from_str(opts).unwrap_or_else(|_| {
            warn!("Invalid JSON for --options; ignoring");
            json!({})
        })
    });

    fs::create_dir_all(output_dir).expect("Failed to create output directory");

    let supported_ext: HashSet<&str> = ["jpg", "jpeg", "png", "bmp", "gif", "webp"]
        .iter()
        .cloned()
        .collect();
    let files: Vec<_> = fs::read_dir(input_dir)
        .expect("Failed to read input directory")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            let ext = path.extension()?.to_str()?.to_lowercase();

            if supported_ext.contains(ext.as_str()) {
                if skip_existing {
                    let json_path = Path::new(output_dir).join(format!(
                        "{}{}.json",
                        path.file_name().unwrap().to_string_lossy().to_string(),
                        file_suffix
                    ));
                    if json_path.exists() {
                        return None;
                    } else {
                        return Some(path);
                    }
                } else {
                    Some(path)
                }
            } else {
                None
            }
        })
        .collect();

    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{wide_bar} {pos}/{len}")
            .expect("Error creating progress bar"),
    );

    let client = Client::new();

    for batch in files.chunks(batch_size) {
        let mut tasks = Vec::new();

        for path in batch {
            let image_b64 = match encode_image_to_base64(path) {
                Ok((b64, _mime)) => b64,
                Err(e) => {
                    error!("Error encoding {}: {}", path.display(), e);
                    pb.inc(1);
                    continue;
                }
            };

            let file_stem = path.file_name().unwrap().to_string_lossy().to_string();
            let client_clone = client.clone();
            let api_url_clone = api_url.clone();
            let model_clone = model.clone();
            let prompt_clone = prompt.clone();
            let schema_clone = schema_obj.clone();
            let options_clone = options.clone();
            let output_dir_clone = output_dir.to_string();
            let file_suffix_clone = file_suffix.clone();
            let pretty_json_clone = pretty_json;

            let task = tokio::spawn(async move {
                match call_ollama_structured(
                    &client_clone,
                    &api_url_clone,
                    &model_clone,
                    &image_b64,
                    &prompt_clone,
                    schema_clone.as_ref(),
                    options_clone.as_ref(),
                )
                .await
                {
                    Ok(resp) => {
                        let content = resp
                            .get("message")
                            .and_then(|m| m.get("content"))
                            .cloned()
                            .unwrap_or(resp.clone());

                        let output_data = if schema_clone.is_some() {
                            serde_json::from_value::<Value>(content.clone()).unwrap_or(content)
                        } else {
                            content
                        };

                        let out_path = Path::new(&output_dir_clone)
                            .join(format!("{}{}.json", file_stem, file_suffix_clone));
                        let mut fo = File::create(&out_path).expect("Failed to create output");

                        let json_val = serde_json::from_str(output_data.as_str().unwrap())
                            .unwrap_or(output_data.clone());

                        let json_str = if pretty_json_clone {
                            serde_json::to_string_pretty(&json_val).unwrap()
                        } else {
                            serde_json::to_string(&json_val).unwrap()
                        };

                        fo.write_all(json_str.as_bytes())
                            .expect("Failed to write output");

                        Ok(())
                    }
                    Err(e) => {
                        error!("Error processing {}: {}", file_stem, e);
                        Err(e)
                    }
                }
            });

            tasks.push(task);
        }

        for task in tasks {
            let _ = task.await;
            pb.inc(1);
        }
    }

    pb.finish_with_message("Done");
}
