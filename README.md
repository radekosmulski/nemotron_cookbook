# Nemotron on Modal

## Deployment

To deploy and run the Nemotron model on Modal, use the following command:

```bash
modal deploy nemotron_inference_modal.py
```

This will deploy the model to Modal's infrastructure and make it available for inference.

## Testing

To test the deployment and see an example of making a request, run:

```bash
modal run nemotron_inference_modal.py
```

This will execute a test request against the deployed model and show you the response.