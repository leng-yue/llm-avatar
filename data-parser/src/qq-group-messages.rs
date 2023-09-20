use std::collections::{HashMap, HashSet};
use std::fs::File;

use clap::Parser;
use log::{info, warn};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::io::{Seek, SeekFrom, Write};
use std::path::PathBuf;

/// Parse QQ group messages into OpenAI jsonl dataset
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// QQ database path, without 1024 offset
    #[arg(short, long, default_value = "data/nt_db")]
    db_path: String,

    /// The user id to parse
    #[arg(short, long)]
    user_id: String,

    /// The password to decrypt the database
    #[arg(short, long)]
    password: String,

    /// The group id to parse, if not specified, parse all groups
    #[arg(short, long)]
    group_id: Option<String>,

    /// History length for each message
    #[arg(short, long, default_value = "50")]
    history_length: usize,

    /// Max number of characters for history
    #[arg(short, long, default_value = "500")]
    max_chars: usize,

    /// Output path
    #[arg(short, long, default_value = "data/qq-group-messages.jsonl")]
    output_path: String,

    /// System prompt
    #[arg(
        short,
        long,
        default_value = "You are {NAME} and you are chatting in a QQ group. "
    )]
    prompt: String,
}

#[derive(Debug, Clone)]
struct Profile {
    qq: i64,
    name: Option<String>,
    unique_id: String,
}

#[derive(Debug, Clone)]
struct GroupMessage {
    msg: String,
    time: i64,
    group_id: String,
    user_id: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct OpenAIMessage {
    role: String,
    content: String,
    name: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct OpenAIDialog {
    messages: Vec<OpenAIMessage>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::new()
        .target(env_logger::Target::Stdout)
        .filter_level(log::LevelFilter::Info)
        .init();

    let args = Args::parse();
    info!("{:?}", args);

    let db_path = PathBuf::from(&args.db_path);

    info!("Building database offsets...");
    let databases = ["profile_info.db", "group_msg_fts.db"];

    // Create .db.offset files by removing 1024 bytes
    for db in databases.iter() {
        let db_path = db_path.join(db);
        let offset_path = db_path.with_extension("db.offset");

        if offset_path.exists() {
            info!("{:?} already exists, skipping...", offset_path.clone());
            continue;
        }

        info!(
            "Processing {:?} -> {:?}",
            db_path.clone(),
            offset_path.clone()
        );

        let mut db_file = File::open(db_path)?;
        let mut offset_file = File::create(offset_path.clone())?;
        db_file.seek(SeekFrom::Start(1024))?;
        std::io::copy(&mut db_file, &mut offset_file)?;

        info!("{:?} created", offset_path.clone());
    }

    info!("Parsing profile_info.db");
    let profile_db = Connection::open(db_path.join("profile_info.db.offset"))?;
    profile_db.pragma(None, "key", &args.password, |_| Ok(()))?;
    profile_db.pragma(None, "kdf_iter", 4000, |_| Ok(()))?;

    let mut stmt = profile_db.prepare("select `1002`,`20002`,`1000` from profile_info_v2")?;
    let profiles: Vec<Profile> = stmt
        .query_map([], |row| {
            Ok(Profile {
                qq: row.get(0)?,
                name: row.get(1)?,
                unique_id: row.get(2)?,
            })
        })?
        .filter(|p| p.is_ok())
        .map(|p| p.unwrap())
        .collect();

    let unique_id_to_profile: HashMap<String, Profile> = profiles
        .clone()
        .into_iter()
        .map(|p| (p.unique_id.clone(), p))
        .collect();
    info!("Found {} profiles", profiles.len());

    // Find groups that the user is in
    let user_unique_id = profiles
        .iter()
        .find(|p| p.qq.to_string() == args.user_id)
        .unwrap()
        .unique_id
        .clone();
    info!("User unique id: {}", user_unique_id);

    info!("Parsing group_msg_fts.db");
    let group_msg_db = Connection::open(db_path.join("group_msg_fts.db.offset"))?;
    group_msg_db.pragma(None, "key", &args.password, |_| Ok(()))?;
    group_msg_db.pragma(None, "kdf_iter", 4000, |_| Ok(()))?;
    let mut stmt =
        group_msg_db.prepare("select `41701`,`40050`,`40021`,`40020` from group_msg_fts")?;
    let group_messages: Vec<GroupMessage> = stmt
        .query_map([], |row| {
            Ok(GroupMessage {
                msg: row.get(0)?,
                time: row.get(1)?,
                group_id: row.get(2)?,
                user_id: row.get(3)?,
            })
        })?
        .filter(|p| p.is_ok())
        .map(|p| p.unwrap())
        .collect();
    info!("Found {} group messages", group_messages.len());

    // Find group messages that the user sent
    let mut user_groups = group_messages
        .iter()
        .filter(|m| m.user_id == user_unique_id)
        .map(|m| m.group_id.clone())
        .collect::<HashSet<String>>();

    info!("User has sent messages in {} groups", user_groups.len());

    if let Some(group_id) = args.group_id.clone() {
        if !user_groups.contains(&group_id) {
            warn!("User has not sent messages in group {}", group_id);
            return Ok(());
        }

        user_groups = HashSet::from([group_id.clone()]);
        info!("Only parsing group {}", group_id);
    }

    let mut counter = 0;

    // Open output file
    let mut output_file = File::create(args.output_path)?;
    let system_prompt = args.prompt.replace("{NAME}", unique_id_to_profile[&user_unique_id].name.as_ref().unwrap_or(&user_unique_id));
    let system_prompt = OpenAIMessage {
        role: "system".to_string(),
        content: system_prompt,
        name: None,
    };

    for group in user_groups {
        let mut group_messages = group_messages
            .iter()
            .filter(|m| m.group_id == group)
            .collect::<Vec<&GroupMessage>>();
        // Sort by time
        group_messages.sort_by(|a, b| a.time.cmp(&b.time));

        // Build message queue
        let mut message_queue = Vec::new();

        for message in group_messages {
            let role = if &message.user_id == &user_unique_id {
                "assistant".to_string()
            } else {
                "user".to_string()
            };

            let name = if unique_id_to_profile.contains_key(&message.user_id) {
                unique_id_to_profile
                    .get(&message.user_id)
                    .unwrap()
                    .name
                    .clone()
                    .unwrap_or(message.user_id.clone())
            } else {
                message.user_id.clone()
            };

            message_queue.push(OpenAIMessage {
                role,
                content: message.msg.clone(),
                name: Some(name),
            });
            message_queue =
                message_queue[message_queue.len().saturating_sub(args.history_length)..].to_vec();

            // Calculate and drop messages that are too long
            let mut message_queue_chars = message_queue
                .iter()
                .map(|m| {
                    m.content.chars().count() + m.name.clone().unwrap_or_default().chars().count()
                })
                .sum::<usize>();
            while message_queue_chars > args.max_chars {
                message_queue_chars -= message_queue[0].content.chars().count();
                message_queue_chars -= message_queue[0]
                    .name
                    .clone()
                    .unwrap_or_default()
                    .chars()
                    .count();
                message_queue = message_queue[1..].to_vec();
            }

            if &message.user_id != &user_unique_id || message_queue.len() == 0 {
                continue;
            }

            // Now save the messages
            let dialog = OpenAIDialog {
                messages: [vec![system_prompt.clone()], message_queue.clone()].concat(),
            };
            let dialog = serde_json::to_string(&dialog)?;
            output_file.write_all(format!("{}\n", dialog).as_bytes())?;

            counter += 1;
            if counter % 1000 == 0 {
                info!("{} messages saved", counter);
            }
        }
    }

    output_file.flush()?;
    info!("Done. {} messages saved", counter);

    Ok(())
}
