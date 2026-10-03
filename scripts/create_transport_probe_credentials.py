"""Create fresh two-day TLS probe credentials; CA private key stays in memory."""
import argparse
import datetime
import os
from pathlib import Path
from cryptography import x509
from cryptography.hazmat.primitives import hashes,serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID,ExtendedKeyUsageOID


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True)
    a=p.parse_args();root=Path(a.directory);root.mkdir(mode=0o700,parents=True,exist_ok=False)
    now=datetime.datetime.now(datetime.timezone.utc)
    ca_key=ec.generate_private_key(ec.SECP256R1())
    ca_name=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,'FedGraph ephemeral probe CA')])
    def builder(subject,key):
        return (x509.CertificateBuilder().subject_name(subject).issuer_name(ca_name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now-datetime.timedelta(minutes=5)).not_valid_after(now+datetime.timedelta(days=2)))
    ca=builder(ca_name,ca_key).add_extension(x509.BasicConstraints(ca=True,path_length=0),critical=True).sign(ca_key,hashes.SHA256())
    (root/'ca.crt').write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    for role,name,purpose in [('server','gtx-coordinator',ExtendedKeyUsageOID.SERVER_AUTH),('client','owner-2',ExtendedKeyUsageOID.CLIENT_AUTH)]:
        key=ec.generate_private_key(ec.SECP256R1())
        subject=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,name)])
        certificate=builder(subject,key).add_extension(x509.BasicConstraints(ca=False,path_length=None),critical=True)
        certificate=certificate.add_extension(x509.ExtendedKeyUsage([purpose]),critical=False)
        if role=='server':certificate=certificate.add_extension(x509.SubjectAlternativeName([x509.DNSName(name)]),critical=False)
        (root/(role+'.crt')).write_bytes(certificate.sign(ca_key,hashes.SHA256()).public_bytes(serialization.Encoding.PEM))
        target=root/(role+'.key')
        with os.fdopen(os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'wb') as f:
            f.write(key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()))
    print('Created public CA and ephemeral role credentials; CA private key not saved')


if __name__=='__main__':main()
