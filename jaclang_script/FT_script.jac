import:py from unsloth, FastVisionModel ;
import:py torch ;
import:py from unsloth, is_bf16_supported ;
import:py from unsloth.trainer, UnslothVisionDataCollator ;
import:py from trl, SFTTrainer, SFTConfig ;
import:py from datasets, load_dataset ;
import:py from bert_score, score ;
import:py numpy as np ;
import:py from transformers, TextStreamer ;

can load_model() {

    (model, tokenizer) = FastVisionModel.from_pretrained(
        'unsloth/Qwen2-VL-7B-Instruct',
        load_in_4bit=True,
        use_gradient_checkpointing='unsloth'
    );    
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias='none',
        random_state=3407,
        use_rslora=False,
        loftq_config=None
    );    
    return (model, tokenizer) ;
}

can prep_train_dataset(num_img: Any) {

    instruction = 'Write the LaTeX representation for this image.';    
    can convert_to_conversation(sample: Any) {

        conversation = [
            {
                'role': 'user',
                'content': [{'type': 'text', 'text': instruction}, {'type': 'image', 'image': sample['image']}]
            },
            {
                'role': 'assistant',
                'content': [{'type': 'text', 'text': sample['text']}]
            }
        ];    
        return {'messages': conversation} ;
    }
    
    dataset = load_dataset('unsloth/LaTeX_OCR', split='train');    
    dataset = dataset.select(range(num_img));    
    converted_dataset = [convert_to_conversation(sample) for sample in dataset];    
    return converted_dataset ;
}

can prep_train_model(model: Any, tokenizer: Any, converted_dataset: Any, num_epochs: Any=5) {

    FastVisionModel.for_training(model) ;    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=converted_dataset,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            #max_steps=30,
            num_train_epochs=num_epochs,
            learning_rate=0.0002,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=1,
            optim='adamw_8bit',
            weight_decay=0.01,
            lr_scheduler_type='linear',
            seed=3407,
            output_dir='outputs',
            report_to='none',
            remove_unused_columns=False,
            dataset_text_field='',
            dataset_kwargs={'skip_prepare_dataset': True},
            dataset_num_proc=4,
            max_seq_length=2048
        )
    );    
    return trainer ;
}

can get_response(test_dataset: Any, model: Any, tokenizer: Any, n: Any) {

    FastVisionModel.for_inference(model) ;    
    image = test_dataset[n]['image'];    
    instruction = 'Write the LaTeX representation for this image.';    
    messages = [
        {
            'role': 'user',
            'content': [{'type': 'image'}, {'type': 'text', 'text': instruction}]
        }
    ];    
    input_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True
    );    
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors='pt'
    ).to('cuda');    
    text_streamer = TextStreamer(tokenizer, skip_prompt=True);    
    output_id = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=128,
        use_cache=True,
        temperature=1.5,
        min_p=0.1  # use `top_p` instead of `min_p`
    );
    response = tokenizer.decode(output_id[0], skip_special_tokens=True);    
    return response ;
}

can evaluate(response: Any, reference: Any) {

    (P, R, F1) = score(
        [response],
        [reference],
        model_type='bert-base-uncased',
        lang='en'
    );    
    return {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
    } ;
}

with entry {

    (model, tokenizer) = load_model();    
    train_dataset = prep_train_dataset(100);    
    trainer = prep_train_model(model, tokenizer, train_dataset);    
    trainer.train() ;         
    test_dataset = load_dataset('unsloth/LaTeX_OCR', split='test');    
    response_dict = {'precision': 0, 'recall': 0, 'f1': 0};    
    
    for img in range(20) {
        response = get_response(test_dataset, model, tokenizer, img);    
        reference = test_dataset[img]['text'];    
        accuracy = evaluate(response, reference);    
        response_dict['precision'] += accuracy['precision'];    
        response_dict['recall'] += accuracy['recall'];    
        response_dict['f1'] += accuracy['f1'];
    }
    
    precision = (response_dict['precision'] / 20);
    recall = (response_dict['recall'] / 20);
    f1 = (response_dict['f1'] / 20);
      
    print(f"{'VQA Accuracy: Precision: '}{precision}{', Recall: '}{recall}{', F1: '}{f1}");
}