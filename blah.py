POST https://www.googleapis.com/compute/v1/projects/voice-research-255602/zones/us-central1-a/instances
{
  "canIpForward": false,
  "confidentialInstanceConfig": {
    "enableConfidentialCompute": false
  },
  "deletionProtection": false,
  "description": "",
  "disks": [
    {
      "autoDelete": true,
      "boot": true,
      "deviceName": "instance-1",
      "initializeParams": {
        "diskSizeGb": "10",
        "diskType": "projects/voice-research-255602/zones/us-central1-a/diskTypes/pd-balanced",
        "labels": {},
        "sourceImage": "projects/debian-cloud/global/images/debian-10-buster-v20220317"
      },
      "mode": "READ_WRITE",
      "type": "PERSISTENT"
    }
  ],
  "displayDevice": {
    "enableDisplay": false
  },
  "guestAccelerators": [
    {
      "acceleratorCount": 4,
      "acceleratorType": "projects/voice-research-255602/zones/us-central1-a/acceleratorTypes/nvidia-tesla-t4"
    }
  ],
  "labels": {},
  "machineType": "projects/voice-research-255602/zones/us-central1-a/machineTypes/n1-standard-1",
  "metadata": {
    "items": []
  },
  "name": "instance-1",
  "networkInterfaces": [
    {
      "accessConfigs": [
        {
          "name": "External NAT",
          "networkTier": "PREMIUM"
        }
      ],
      "subnetwork": "projects/voice-research-255602/regions/us-central1/subnetworks/default"
    }
  ],
  "reservationAffinity": {
    "consumeReservationType": "ANY_RESERVATION"
  },
  "scheduling": {
    "automaticRestart": false,
    "onHostMaintenance": "TERMINATE",
    "preemptible": false
  },
  "serviceAccounts": [
    {
      "email": "577841675356-compute@developer.gserviceaccount.com",
      "scopes": [
        "https://www.googleapis.com/auth/devstorage.read_only",
        "https://www.googleapis.com/auth/logging.write",
        "https://www.googleapis.com/auth/monitoring.write",
        "https://www.googleapis.com/auth/servicecontrol",
        "https://www.googleapis.com/auth/service.management.readonly",
        "https://www.googleapis.com/auth/trace.append"
      ]
    }
  ],
  "shieldedInstanceConfig": {
    "enableIntegrityMonitoring": true,
    "enableSecureBoot": false,
    "enableVtpm": true
  },
  "tags": {
    "items": []
  },
  "zone": "projects/voice-research-255602/zones/us-central1-a"
}
