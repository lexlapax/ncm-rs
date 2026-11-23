//! Multi-threaded HTTP/1.1 listener with graceful shutdown for `ncm-serve`.
use crate::{
    InferenceEngine, api,
    http::{ParseError, read_request, write_json},
};
use serde_json::json;
use std::{
    io,
    net::{SocketAddr, TcpListener, TcpStream},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

/// Handle used to signal graceful shutdown of a running [`Server`].
#[derive(Clone)]
pub struct ServerHandle {
    shutdown: Arc<AtomicBool>,
    addr: SocketAddr,
}

impl ServerHandle {
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
        // Nudge the listener out of accept if it is waiting.
        let _ = TcpStream::connect(self.addr);
    }
}

/// Pure-std HTTP/1.1 server exposing OpenAI (`/v1/*`) and Ollama (`/api/*`) endpoints.
pub struct Server<E: InferenceEngine> {
    listener: TcpListener,
    engine: Arc<E>,
    shutdown: Arc<AtomicBool>,
}

impl<E: InferenceEngine> Server<E> {
    pub fn bind(addr: &str, engine: E) -> io::Result<Self> {
        let listener = TcpListener::bind(addr)?;
        listener.set_nonblocking(false)?;
        Ok(Self {
            listener,
            engine: Arc::new(engine),
            shutdown: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.listener.local_addr()
    }

    pub fn handle(&self) -> io::Result<ServerHandle> {
        Ok(ServerHandle {
            shutdown: Arc::clone(&self.shutdown),
            addr: self.listener.local_addr()?,
        })
    }

    /// Accept connections until [`ServerHandle::shutdown`] is signaled.
    pub fn serve(self) -> io::Result<()> {
        let _ = self.listener.set_nonblocking(true);
        while !self.shutdown.load(Ordering::SeqCst) {
            match self.listener.accept() {
                Ok((mut stream, _peer)) => {
                    if self.shutdown.load(Ordering::SeqCst) {
                        break;
                    }
                    let _ = stream.set_nonblocking(false);
                    let engine = Arc::clone(&self.engine);
                    thread::spawn(move || {
                        handle_connection(&*engine, &mut stream);
                    });
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(10));
                }
                Err(_) if self.shutdown.load(Ordering::SeqCst) => break,
                Err(_) => {
                    thread::sleep(Duration::from_millis(25));
                }
            }
        }
        Ok(())
    }
}

fn handle_connection<E: InferenceEngine>(engine: &E, stream: &mut TcpStream) {
    match read_request(stream) {
        Ok(req) => {
            let _ = api::dispatch(engine, &req, stream);
        }
        Err(ParseError::Eof) => {}
        Err(ParseError::BadRequest(msg)) => {
            let _ = write_json(
                stream,
                400,
                &json!({ "error": { "message": msg, "type": "invalid_request_error" } }),
            );
        }
        Err(ParseError::PayloadTooLarge(msg)) => {
            let _ = write_json(
                stream,
                413,
                &json!({ "error": { "message": msg, "type": "invalid_request_error" } }),
            );
        }
        Err(ParseError::Io) => {}
    }
}
