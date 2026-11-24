import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/ort.webgpu.min.mjs';
const out=document.getElementById('out');const log=(...a)=>{out.textContent+='\n'+a.join(' ')};
ort.env.wasm.numThreads=1;ort.env.wasm.wasmPaths='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/';
const mb=n=>(n/1048576).toFixed(0)+' MB';
document.getElementById('b1').onclick=async()=>{
  out.textContent='Device check';
  log('userAgent:',navigator.userAgent);
  log('deviceMemory (GB, capped by browser):',navigator.deviceMemory??'not reported');
  log('cpu threads:',navigator.hardwareConcurrency);
  log('secure context:',window.isSecureContext);
  try{const e=await navigator.storage.estimate();log('storage free approx:',mb(e.quota-e.usage))}catch(e){log('storage estimate: n/a')}
  if(!navigator.gpu){log('WebGPU: NOT available in this browser');return}
  try{const a=await navigator.gpu.requestAdapter();
    if(!a){log('WebGPU: no adapter returned');return}
    const i=a.info||{};log('WebGPU adapter:',[i.vendor,i.architecture,i.description].filter(Boolean).join(' / ')||'(hidden)');
    const l=a.limits;log('maxBufferSize:',mb(l.maxBufferSize),' maxStorageBufferBindingSize:',mb(l.maxStorageBufferBindingSize));
  }catch(e){log('WebGPU error:',e.message)}
};
document.getElementById('b2').onclick=async()=>{
  log('\nDownloading decoder...');
  const t=performance.now();
  try{
    const url='https://huggingface.co/ipsilondev/chatterbox-multilingual-ONNX-q4/resolve/main/onnx/conditional_decoder.onnx';
    const r=await fetch(url);if(!r.ok)throw new Error('HTTP '+r.status);
    const buf=new Uint8Array(await r.arrayBuffer());
    log('downloaded',mb(buf.length),'in',((performance.now()-t)/1000).toFixed(0),'s');
    const t2=performance.now();
    const s=await ort.InferenceSession.create(buf,{executionProviders:['webgpu']});
    log('decoder session created on WebGPU in',((performance.now()-t2)/1000).toFixed(1),'s; inputs:',s.inputNames.join(','));
    if(performance.memory)log('JS heap used:',mb(performance.memory.usedJSHeapSize));
    log('RESULT: decoder loads. (Not a full audio test.)');
  }catch(e){log('RESULT: failed -',e.message||e)}
};
