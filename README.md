# Webová aplikácia text na obrázok

Tento projekt je súčasťou diplomovej práce Adriana Komorného, jedná sa o webovú aplikáciu txt2img, ktorá dokáže z textu zhotovit obrázok. Táto webová aplikácia je vyvíjaná pomocou Angularu, Spring Boot, Dockeru a docker-compose.

## Požiadavky

Pred spustením tejto aplikácie je potrebné mať nainštalované nasledujúce nástroje:
- Docker
- docker-compose

## Inštalácia a spustenie

1. Stiahnite si zdrojový kód z tohto repozitára: https://github.com/adrian560/Dpfiles.git

2. Prejdite do vnútra repozitára

3. Pry prvom spustení použite príkaz: docker-compose up --build
Po vykonaní príkazu sa spustí kontajner s aplikáciou. Pri prvom spustení to môže trvať niekoľko minút

4. Po úspešnom spustení aplikácie môžete v prehliadači otvoriť [http://localhost:4300](http://localhost:4300) a zobraziť sa vám webová aplikácia.

## Spustenie neskôr

5. Ak ste už raz aplikáciu skompilovali a spustili, môžete ju neskôr spustiť jednoducho pomocou: docker-compose up
Tento príkaz spustí kontajner s aplikáciou na pozadí.(mus=i byť už vybuildený)

## Koniec

Gratulujem! Podarilo sa vám spustit webová aplikácia txt2img

## Linky
Môže sa stať že vám niektoré priečinky nebudú fungovať. V takom prípade sa obráte na nasledujúce linky:

https://github.com/AbdBarho/stable-diffusion-webui-docker.git

